"""
Patient identity, login, and biometric storage for MIRA.

Replaces the old standalone registration_app.py / db.py / face_utils.py
trio. There's now exactly one place patients live: this module's SQLite
table, keyed by name + password for login. Everything camera-related is
handled entirely by pipeline.py (the same burst-capture flow used for a
normal scan) -- mira_app.py hands this module the photo pipeline.py already
saved, and this module turns it into a face encoding ("digital fingerprint")
with face_recognition:

    - Sign up: form details + password are collected, then the very same
      capture used for a scan also produces the photo this module encodes
      and stores as that patient's fingerprint.
    - Log in: name + password identify *which* patient's fingerprint to
      check against; a fresh capture from the same flow is compared to it
      with compare_faces(). A mismatch means "wrong person," not "wrong
      password" -- the password only looks up whose fingerprint to use.

extract_face_encoding() and match_encoding() need opencv-python and
face_recognition installed (see registration_requirements.txt). Everything
else here (SQLite, password hashing) is stdlib + security.py only, so
signup/login and the DB itself still work even before those are installed
-- only the actual face capture/verification step needs them.
"""

import hashlib
import json
import os
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime

import numpy as np

import security

try:
    import face_recognition
    LIBS_AVAILABLE = True
except ImportError:
    LIBS_AVAILABLE = False

# Same Database/ convention as pipeline.py: one level up from src/.
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.join(os.path.dirname(SRC_DIR), "Database")
os.makedirs(DATABASE_DIR, exist_ok=True)

DB_PATH = os.path.join(DATABASE_DIR, "mira_patients.db")

MATCH_TOLERANCE = 0.5  # lower = stricter. face_recognition's default is 0.6.


class FaceMatchError(Exception):
    """Raised when a captured photo can't be turned into a face encoding
    (no face found where pipeline.py should already have found one, or the
    libraries aren't installed)."""


SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,

    patient_id                  TEXT UNIQUE NOT NULL,   -- internal folder/history ID, passed to pipeline.py as --id
    name                        TEXT NOT NULL,
    name_key                    TEXT UNIQUE NOT NULL,   -- normalized name, used as the login key
    password_hash               TEXT NOT NULL,

    age                         INTEGER NOT NULL,
    sex                         TEXT NOT NULL,
    guardian_name               TEXT,                   -- Spouse / Father
    address                     TEXT,
    contact_number              TEXT,

    registration_fee            REAL DEFAULT 0,
    fee_paid                    INTEGER DEFAULT 0,

    blood_group                 TEXT,
    known_allergies             TEXT,
    existing_conditions         TEXT,
    emergency_contact_name      TEXT,
    emergency_contact_number    TEXT,

    face_encoding                TEXT NOT NULL,          -- JSON 128-d vector ("digital fingerprint")
    photo_path                   TEXT,

    created_at                   TEXT NOT NULL
);
"""


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_connection() as conn:
        conn.execute(SCHEMA)


# ─────────────────────────────────────────────────────────────────────────
# Identity helpers
# ─────────────────────────────────────────────────────────────────────────

def normalize_name(name: str) -> str:
    value = str(name).strip().casefold()
    return re.sub(r"\s+", " ", value)


def generate_patient_id() -> str:
    """A short internal ID, generated once at signup and reused for every
    later visit -- this is what keeps a patient's scan history/trends
    continuous, and is what gets passed to pipeline.py as --id. Never
    typed by the patient; login is by name + password instead."""
    return f"{time.strftime('%y%m%d')}-{uuid.uuid4().hex[:6].upper()}"


def name_exists(name: str) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM patients WHERE name_key = ?", (normalize_name(name),)
        ).fetchone()
        return row is not None


def get_by_name(name: str):
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM patients WHERE name_key = ?", (normalize_name(name),)
        ).fetchone()


def get_by_patient_id(patient_id: str):
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM patients WHERE patient_id = ?", (patient_id,)
        ).fetchone()


def check_login(name: str, password: str):
    """Returns the patient row if name+password are valid, else None."""
    patient = get_by_name(name)
    if not patient:
        return None
    if not security.verify_password(password, patient["password_hash"]):
        return None
    return patient


# ─────────────────────────────────────────────────────────────────────────
# Face encoding ("digital fingerprint")
# ─────────────────────────────────────────────────────────────────────────

def _require_libs():
    if not LIBS_AVAILABLE:
        raise FaceMatchError(
            "opencv-python and face_recognition are not installed.\n"
            "Run: pip install -r registration_requirements.txt"
        )


def extract_face_encoding(photo_path: str) -> np.ndarray:
    """Turns a saved photo (the one pipeline.py's burst capture already
    picked and saved) into a 128-d face encoding. pipeline.py only saves a
    frame once its own face detector found exactly one clear face in it,
    so this should almost always succeed -- but a second, independent
    detector (face_recognition's) is still doing the actual detection
    here, so it's not guaranteed."""
    _require_libs()

    image = face_recognition.load_image_file(photo_path)
    locations = face_recognition.face_locations(image)

    if len(locations) == 0:
        raise FaceMatchError(
            "No face could be found in the captured photo. Please try the "
            "capture again."
        )
    if len(locations) > 1:
        raise FaceMatchError(
            "More than one face was found in the captured photo. Only the "
            "patient should be in frame during capture."
        )

    encodings = face_recognition.face_encodings(image, known_face_locations=locations)
    return encodings[0]


def encoding_to_json(encoding: np.ndarray) -> str:
    return json.dumps(np.asarray(encoding).tolist())


def encoding_from_json(payload: str) -> np.ndarray:
    return np.array(json.loads(payload))


def match_encoding(candidate_encoding, stored_encoding_json, tolerance: float = MATCH_TOLERANCE) -> bool:
    """Compares a freshly-captured encoding against a patient's stored
    fingerprint. True = same person."""
    _require_libs()
    stored = encoding_from_json(stored_encoding_json)
    results = face_recognition.compare_faces([stored], candidate_encoding, tolerance=tolerance)
    return bool(results[0])


# ─────────────────────────────────────────────────────────────────────────
# Insert (signup)
# ─────────────────────────────────────────────────────────────────────────

def insert_patient(
    patient_id,
    name,
    password_hash,
    age,
    sex,
    guardian_name,
    address,
    contact_number,
    registration_fee,
    fee_paid,
    blood_group,
    known_allergies,
    existing_conditions,
    emergency_contact_name,
    emergency_contact_number,
    face_encoding,
    photo_path,
):
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO patients (
                patient_id, name, name_key, password_hash,
                age, sex, guardian_name, address, contact_number,
                registration_fee, fee_paid,
                blood_group, known_allergies, existing_conditions,
                emergency_contact_name, emergency_contact_number,
                face_encoding, photo_path, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                patient_id, name, normalize_name(name), password_hash,
                age, sex, guardian_name, address, contact_number,
                registration_fee, int(fee_paid),
                blood_group, known_allergies, existing_conditions,
                emergency_contact_name, emergency_contact_number,
                face_encoding, photo_path, datetime.utcnow().isoformat(),
            ),
        )
    return get_by_patient_id(patient_id)
