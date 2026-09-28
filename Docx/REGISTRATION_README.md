# MIRA Patient Registry — Login / Signup (v1)

A desktop Tkinter app, styled to match MIRA's existing dark theme, for
registering patients and logging them in. Data is stored locally in SQLite
(`mira_patients.db`).

## What it stores right now

- **Identity:** name, age, sex, spouse/father's name, address, contact number
- **Medical:** blood group, known allergies, existing conditions
- **Emergency contact:** name and number
- **Administrative:** registration fee, fee-paid status, registration timestamp
- **Biometric:** the captured photo, its face encoding (128-d vector), and the
  `unique_id` derived from it
- **Account:** an auto-generated `login_id` and a hashed password (PBKDF2,
  stdlib only — no extra dependency for this part)

Two identifiers, on purpose:
- `unique_id` (e.g. `PT-4F2A9C1B03`) — derived from the registration details
  (name, age, sex, registration fee, spouse/father's name, address), not
  from the face.
- `login_id` (e.g. `johnsm482`) — what the patient actually types in to log
  in, alongside their password.

The face is used for one thing only: catching the same person being
registered twice. It's compared against every photo already on file
(`find_matching_patient`) — no feature vector is hashed into an ID, so it
never acts as a biometric "fingerprint" for identity.

## Setup

Needs a webcam on the machine you run this on, and `dlib` underneath
`face_recognition` (the heaviest install step — it's a C++ library).

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# dlib needs CMake + a C++ compiler first:
#   macOS:   brew install cmake
#   Ubuntu:  sudo apt install cmake build-essential
#   Windows: install "Desktop development with C++" via Visual Studio Build Tools

pip install -r requirements.txt
python app.py
```

Everything else (Tkinter, sqlite3, hashlib) is in the Python standard
library — nothing else to install.

## How it works

1. **Log In** screen opens first — Login ID + password.
2. **Register a New Patient** opens a scrollable intake form (identity,
   medical, emergency contact, account/password).
3. On **Capture Photo & Register**, the form is validated, then the app
   switches to a progress screen and opens the webcam on a background
   thread (so the UI doesn't freeze) — takes one photo, detects the face.
4. That face is checked against everyone already registered. A match blocks
   the new registration and shows the existing patient's ID instead.
5. Otherwise: `unique_id` is derived from the registration details, `login_id`
   is generated from the name, the password is hashed, and the record is
   saved. The new IDs are shown once on a confirmation screen, then the
   full record on the dashboard.

This assumes the webcam is attached to the same machine the app runs on —
a reception-desk/kiosk setup, same as MIRA's existing scan flow.

## Before using this with real patients

- **This is a v1** — good for prototyping the registration flow, not yet
  hardened for production use.
- Face encodings and photos are biometric data, and blood group/allergies/
  conditions are medical data. Depending on where you operate this may carry
  legal obligations (HIPAA in the US, India's DPDP Act, GDPR in the EU, etc.)
  — worth checking what consent and retention rules apply before deploying.
- `mira_patients.db` and `captured_faces/` are stored unencrypted on disk.
  For real use, put this behind disk-level encryption and restrict file
  permissions at minimum.
- There's no admin/staff role yet — this is single-patient-at-a-time,
  matching what you asked for first. Say if you want a separate staff login
  that can search/list every patient.

## Next steps (tell me what you want and I'll build it)

- Admin/staff dashboard to search and list all patients
- Editing a record after registration
- Logging in by face instead of password
- Wiring this into MIRA's existing `pipeline.py` / `Database/` folder
  structure, so registration and the analysis scan share one patient record
- Payment tracking / receipts for the registration fee
- Exporting records (CSV/PDF)
