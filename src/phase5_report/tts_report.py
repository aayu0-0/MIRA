"""
Phase 5 — TTS Report (File 4 of 4)
--------------------------------------
Generates a plain-text file optimized for text-to-speech (TTS) and LLM
simplification pipelines (OllamaVoice / Piper TTS).

Design decisions:
- No symbols: no %, °, /, |, bullets, brackets, dashes used as punctuation
- No raw scores or metric names — descriptions only
- Natural spoken rhythm: short sentences, no subordinate clause chains
- Severity expressed in plain speech: "a little", "noticeably", "significantly"
- Paragraph breaks = natural pause points for TTS
- No headers with colons that read awkwardly aloud
- Quality warning rewritten as a spoken caveat, not a banner
- Ends with a short spoken summary the user can act on
- File is UTF-8 plain text, no markdown, no XML tags
- Addresses the patient by name, never by numeric ID
"""

import datetime
import re
from pathlib import Path


# Plain-speech substitutions for jargon that reads awkwardly aloud
_REPLACEMENTS = [
    ("texture irregularity",  "uneven skin texture"),
    ("pallor",                "paleness"),
    ("erythema",              "redness"),
    ("asymmetry",             "unevenness"),
    ("puffiness",             "puffiness under the eye"),
    ("dark circle",           "dark area under the eye"),
    ("openness ratio",        "how open the eye appears"),
    ("acne score",            "breakout level"),
    ("color consistency",     "colour evenness"),
    ("curvature score",       "mouth curvature"),
    ("aspect ratio",          "face shape"),
    ("swelling score",        "puffiness around the face"),
    ("alignment angle",       "head tilt"),
    ("symmetry score",        "face balance"),
    ("notable",               "significant"),
    ("moderate",              "noticeable"),
    (" mild ",                " slight "),
    (" / ",                   " or "),
]

# Report-writer labels that read fine on paper but are dead weight spoken aloud.
# Stripped entirely rather than replaced, so the sentence that follows just
# flows as plain speech.
_LABEL_PREFIXES = [
    "observation:", "possible indicator:", "possible indicator of",
    "visible sign:", "note:", "indicator:", "finding:",
]

# Anything containing a raw number in parentheses is score leakage
# (e.g. "(score 89/100)", "(left 81/100, right 96/100)", "(2 area(s) detected)")
# Allows one level of nested parens, e.g. "(2 area(s) detected)".
_PAREN_WITH_DIGIT_RE = re.compile(
    r"\s*\((?:[^()]|\([^()]*\))*\d(?:[^()]|\([^()]*\))*\)"
)

# Bare "89/100", "18.1", "100.0", "X percent" style fragments that slip in
# outside parentheses
_RAW_SCORE_RE = re.compile(
    r"\b\d+(\.\d+)?\s*(/\s*100|percent|%)?\b(?=[\s.,;]|$)"
)

# Collapse doubled whitespace/punctuation left behind after stripping
_WS_RE = re.compile(r"\s{2,}")
_DANGLING_PUNCT_RE = re.compile(r"\s+([.,;])")


def _strip_labels(text: str) -> str:
    lowered = text.strip()
    for label in _LABEL_PREFIXES:
        if lowered.lower().startswith(label):
            lowered = lowered[len(label):].strip()
            break
    return lowered


def _strip_scores(text: str) -> str:
    text = _PAREN_WITH_DIGIT_RE.sub("", text)
    text = _RAW_SCORE_RE.sub("", text)
    return text


def _tidy_whitespace(text: str) -> str:
    text = _WS_RE.sub(" ", text)
    text = _DANGLING_PUNCT_RE.sub(r"\1", text)
    return text.strip()


def _clean(text: str, capitalize: bool = True) -> str:
    """Apply spoken-language substitutions to a finding string.

    capitalize: capitalise the first letter (correct for the English
        script). The Hindi builder passes False, since Devanagari has no
        case and capitalising would just leave a stray uppercase letter
        sitting in the middle of leftover, not-yet-covered English text.
    """
    text = _strip_labels(text)
    for src, dst in _REPLACEMENTS:
        text = text.replace(src, dst)
    # Strip bracket tags that may leak in
    text = re.sub(r"\[.*?\]", "", text)
    # Strip raw scores/ratios/percentages — spoken reports describe, don't cite numbers
    text = _strip_scores(text)
    text = _tidy_whitespace(text)
    if capitalize and text:
        text = text[0].upper() + text[1:]
    return text


def _period(text: str) -> str:
    """Ensure text ends with a period for natural TTS pause."""
    text = text.rstrip(" .!?")
    # Already ends with a Hindi sentence-ending danda -- don't also add "."
    if text.endswith("।"):
        return text
    return text + "." if text else ""


def _join_findings(findings: list, connector_first: str = "", connectors: list = None) -> str:
    """
    Join a list of already-cleaned finding sentences with natural spoken
    connectors instead of just concatenating periods back to back, so a list
    of two or three findings doesn't sound like a bulleted report.
    """
    sentences = [_period(_clean(f)) for f in findings if _clean(f)]
    sentences = [s for s in sentences if s]
    if not sentences:
        return ""
    if len(sentences) == 1:
        return sentences[0]

    connectors = connectors or ["Also,", "On top of that,", "As well,"]
    out = [sentences[0]]
    for i, s in enumerate(sentences[1:]):
        connector = connectors[i % len(connectors)]
        # lowercase the first letter since it now follows a connector
        s_lower = s[0].lower() + s[1:] if s else s
        out.append(f"{connector} {s_lower}")
    return " ".join(out)


def _lowered(text: str) -> str:
    """Lowercase just the first character, so a joined finding string reads
    naturally after a lowercase lead-in phrase like 'For your eyes, ...'."""
    return text[0].lower() + text[1:] if text else text


def _observation_score_caveat(score: float) -> str:
    """
    Only mention the observation score when it's low enough to matter -- a
    high-scoring reading doesn't need a hedge, and skipping it keeps the
    report shorter and more encouraging.
    """
    score = round(score)
    if score >= 75:
        return ""
    if score >= 50:
        return "This reading has a moderate observation score, so treat it as a helpful first look."
    return "This reading has a lower observation score, so it's worth retaking the photo for a clearer picture."


# ── Native Hindi scaffold (hand-written, not machine-translated) ───────────
# Mirrors the English scaffold in _build() one-for-one. Kept as real Hindi
# sentences from the start rather than English sentences run through a
# find-and-replace pass, so section intros/closings never end up half
# translated just because a particular finding sentence wasn't covered by
# hindi_translation.py's dictionaries.

_HI_CONNECTORS = ["साथ ही,", "इसके अलावा,", "और,"]

_HI_SPOKEN_LABEL = {
    "No Major Visible Abnormalities":
        "आप बहुत अच्छे हैं, कोई चिंता की बात नहीं दिखी",
    "Mild Observations Present":
        "आप कुल मिलाकर ठीक हैं, बस एक-दो छोटी बातों पर थोड़ा ध्यान देने की ज़रूरत है",
    "Moderate Observations Present":
        "कुछ बातों पर नज़र रखने लायक हैं, पर इनमें से किसी से घबराने की ज़रूरत नहीं है",
    "Notable Observations Present":
        "कुछ बातें सामने आई हैं जिन्हें जांच करवाना अच्छा रहेगा, और इन्हें जल्दी पकड़ना एक अच्छा कदम है",
}


def _observation_score_caveat_hi(score: float) -> str:
    score = round(score)
    if score >= 75:
        return ""
    if score >= 50:
        return "इस बार फोटो से पूरी तरह साफ़-साफ़ जानकारी नहीं मिल पाई, तो इसे बस एक हल्की झलक जैसा ही समझें।"
    return "इस बार फोटो से ठीक से पहचान नहीं हो पाई, तो अच्छी रोशनी में एक और फोटो लेना बेहतर रहेगा।"


def _join_findings_hi(findings: list) -> str:
    """
    Hindi counterpart to _join_findings(): cleans + translates each finding
    INDIVIDUALLY (via hindi_translation.translate_finding), then joins with
    native Hindi connectors. Never runs translation over the joined string.
    """
    from phase5_report.hindi_translation import translate_finding, is_pure_hindi

    sentences = []
    for f in findings:
        s = _period(translate_finding(_clean(f, capitalize=False)))
        # Drop anything that still has leftover Latin script rather than
        # speaking a mixed-language sentence -- a Hindi TTS voice reads
        # that as dead air / a dropped segment, not as English.
        if s and is_pure_hindi(s):
            sentences.append(s)
    sentences = [s for s in sentences if s]
    if not sentences:
        return ""
    if len(sentences) == 1:
        return sentences[0]

    out = [sentences[0]]
    for i, s in enumerate(sentences[1:]):
        out.append(f"{_HI_CONNECTORS[i % len(_HI_CONNECTORS)]} {s}")
    return " ".join(out)


def _section_hi(intro: str, joined: str) -> str:
    """Prefix a section intro onto its joined findings, but only if there's
    actually something left to say -- avoids a dangling intro phrase with
    nothing after it when every finding in the section got filtered out
    for still containing Latin script."""
    return f"{intro} {joined}" if joined else ""


class TTSReportGenerator:

    def generate(self,
                 p1, p3, p4,
                 quality_warnings: list = None,
                 patient_name: str = None,
                 patient_id: str = "UNKNOWN",
                 output_dir: str = None,
                 language: str = "both") -> dict:
        """
        Build the TTS plain-text report.

        patient_name: how the patient should be addressed in the spoken
            report, e.g. "Aayush Sardana". This is what gets spoken aloud.
        patient_id: internal identifier, used only for the output filename,
            never spoken in the report text.
        language: "both" (default), "en", or "hi".
            - "both": builds the English script (_build) and the Hindi
              script (_build_hi) independently from the SAME p1/p3/p4
              data -- the Hindi text is natively assembled from hand-written
              Hindi scaffolding, not produced by translating the finished
              English paragraph. Only the dynamic per-finding sentences are
              run through hindi_translation.translate_finding, one finding
              at a time (before joining), which avoids the substring-
              collision bugs that whole-paragraph translation caused (e.g.
              "significant" matching inside "insignificant"). Both scripts
              are returned, and both are written to disk when output_dir is
              given (tts_report_{id}.txt and tts_report_{id}_hi.txt).
            - "en": English only.
            - "hi": Hindi only, still built natively via _build_hi.
            Any other value is treated as "both".

        Returns:
            {
                "text":      str,        # primary script (English unless language=="hi")
                "path":      str | None, # primary written file path
                "language":  str,        # language of the primary "text"/"path" ("en" or "hi")
                "text_en":   str | None, # English script, when built
                "path_en":   str | None,
                "text_hi":   str | None, # Hindi script, when built
                "path_hi":   str | None,
            }
        """
        quality_warnings = quality_warnings or []
        if language not in ("en", "hi", "both"):
            language = "both"

        text_en = self._build(p1, p3, p4, quality_warnings, patient_name, patient_id)

        text_hi = None
        if language in ("hi", "both"):
            text_hi = self._build_hi(p1, p3, p4, quality_warnings, patient_name, patient_id)

        path_en = None
        path_hi = None
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            if language in ("en", "both"):
                path_en = f"{output_dir}/tts_report_{patient_id}.txt"
                Path(path_en).write_text(text_en, encoding="utf-8")
            if language in ("hi", "both"):
                path_hi = f"{output_dir}/tts_report_{patient_id}_hi.txt"
                Path(path_hi).write_text(text_hi, encoding="utf-8")

        # Primary text/path: English by default, Hindi only if that's all
        # that was requested.
        if language == "hi":
            primary_text, primary_path, primary_lang = text_hi, path_hi, "hi"
        else:
            primary_text, primary_path, primary_lang = text_en, path_en, "en"

        return {
            "text": primary_text,
            "path": primary_path,
            "language": primary_lang,
            "text_en": text_en if language in ("en", "both") else None,
            "path_en": path_en,
            "text_hi": text_hi,
            "path_hi": path_hi,
        }

    # ── Builder ───────────────────────────────────────────────────────────────

    def _build(self, p1, p3, p4, quality_warnings, patient_name, patient_id) -> str:
        now = datetime.datetime.now().strftime("%d %B %Y")
        blocks = []

        # ── Opening ──────────────────────────────────────────────────────────
        name = (patient_name or "").strip()
        greeting = f"Hi {name}, here" if name else "Here"

        # ── Quality caveat ────────────────────────────────────────────────────
        caveat_block = None
        if quality_warnings:
            caveat_block = (
                "Before we begin, a quick note on photo quality. "
                + " ".join(quality_warnings)
                + " A retake in good lighting, facing the camera, would help next time."
            )

        # ── Overall result ────────────────────────────────────────────────────
        label = p4.overall_label

        spoken_label = {
            "No Major Visible Abnormalities": "you're doing great, nothing concerning stood out",
            "Mild Observations Present":      "you're doing well overall, with just a couple of small things worth a little attention",
            "Moderate Observations Present":  "there are a few things worth keeping an eye on, though nothing here should worry you",
            "Notable Observations Present":   "a few things stood out that are worth getting checked, and catching them early is a good move",
        }.get(label, label.lower())

        opening = f"{greeting} is your facial health summary for {now}. Overall, {spoken_label}."
        conf_note = _observation_score_caveat(p4.observation_score)
        if conf_note:
            opening += f" {conf_note}"
        blocks.append(opening)

        if caveat_block:
            blocks.append(caveat_block)

        if p4.overall:
            blocks.append(_join_findings(p4.overall))

        # ── Face balance ──────────────────────────────────────────────────────
        face_lines = [t for t in [
            p4.face.symmetry, p4.face.alignment,
            p4.face.proportion, p4.face.swelling
        ] if t]
        if face_lines:
            face_text = _join_findings(face_lines)
            face_text = face_text[0].lower() + face_text[1:] if face_text else face_text
            blocks.append("For your face shape and balance, " + face_text)

        # ── Eyes ──────────────────────────────────────────────────────────────
        eye_obs = p4.eyes.left + p4.eyes.right
        non_normal_eyes = [o for o in eye_obs if not o.is_normal]
        if not non_normal_eyes:
            blocks.append("Your eyes look healthy, no signs of tiredness, puffiness, or redness.")
        else:
            blocks.append("For your eyes, " + _lowered(_join_findings([o.finding for o in non_normal_eyes])))

        if p4.eyes.openness_note:
            cleaned_note = _period(_clean(p4.eyes.openness_note))
            if cleaned_note:
                blocks.append(cleaned_note)

        # ── Skin ──────────────────────────────────────────────────────────────
        skin_obs = (
            p4.skin.forehead + p4.skin.left_cheek +
            p4.skin.right_cheek + p4.skin.nose + p4.skin.overall
        )
        non_normal_skin = [o for o in skin_obs if not o.is_normal]
        if not non_normal_skin:
            blocks.append("Your skin looks clear, no significant breakouts, redness, or uneven texture.")
        else:
            blocks.append("On your skin, " + _lowered(_join_findings([o.finding for o in non_normal_skin])))

        # ── Lips ──────────────────────────────────────────────────────────────
        if not p3.lips.available:
            blocks.append("Lip analysis wasn't available from this image.")
        else:
            non_normal_lips = [o for o in p4.lips.findings if not o.is_normal]
            if not non_normal_lips:
                blocks.append("Your lips look healthy, no signs of dryness or paleness.")
            else:
                blocks.append("For your lips, " + _lowered(_join_findings([o.finding for o in non_normal_lips])))

        # ── Mouth ─────────────────────────────────────────────────────────────
        if p3.smile.available:
            non_normal_mouth = [o for o in p4.mouth.findings if not o.is_normal]
            if not non_normal_mouth:
                blocks.append("Your mouth shape looks balanced and even.")
            else:
                blocks.append("For your mouth shape, " + _lowered(_join_findings([o.finding for o in non_normal_mouth])))

        # ── Closing ───────────────────────────────────────────────────────────
        all_obs = (
            p4.eyes.left + p4.eyes.right +
            p4.skin.forehead + p4.skin.left_cheek +
            p4.skin.right_cheek + p4.skin.nose + p4.skin.overall +
            p4.lips.findings + p4.mouth.findings
        )
        non_normal = [o for o in all_obs if not o.is_normal]
        notable    = [o for o in non_normal
                      if o.severity.value.lower() in ("moderate", "notable")]

        sign_off = f", {name}" if name else ""

        if not non_normal:
            closing = (
                f"That wraps up your summary{sign_off}. Everything looks good, keep it up. "
                "Check back in a few days to keep tracking your progress."
            )
        elif notable:
            closing = (
                f"That wraps up your summary{sign_off}. A few things are worth getting checked by a "
                "professional, but catching them early like this is a good move. "
                "This isn't a diagnosis, just a helpful nudge to look after yourself."
            )
        else:
            closing = (
                f"That wraps up your summary{sign_off}. A few small things showed up, but nothing "
                "to worry about. These often ease up with rest and hydration. "
                "Try again in a few days to see how things are tracking."
            )
        blocks.append(closing)

        # Join blocks with double newline (paragraph break = TTS pause cue)
        return "\n\n".join(b for b in blocks if b) + "\n"

    # ── Hindi builder ────────────────────────────────────────────────────────
    # Builds the Hindi script natively from the same p1/p3/p4 data _build()
    # uses -- it is not a translation of the English text. Scaffold lines
    # (greeting, section intros, closings) are hand-written Hindi; only the
    # dynamic per-finding sentences go through hindi_translation.translate_finding,
    # one finding at a time (see _join_findings_hi).

    def _build_hi(self, p1, p3, p4, quality_warnings, patient_name, patient_id) -> str:
        from phase5_report.hindi_translation import translate_finding, is_pure_hindi, _translate_dates

        now = datetime.datetime.now().strftime("%d %B %Y")
        now_hi = _translate_dates(now)
        blocks = []

        # ── Opening ──────────────────────────────────────────────────────────
        name = (patient_name or "").strip()
        greeting = (
            f"नमस्ते {name}, यह रहा आपका चेहरे की सेहत का सारांश"
            if name else "यह रहा आपका चेहरे की सेहत का सारांश"
        )

        # ── Quality caveat ───────────────────────────────────────────────────
        caveat_block = None
        if quality_warnings:
            warning_sentences = []
            for w in quality_warnings:
                s = _period(translate_finding(_clean(w, capitalize=False)))
                if s and is_pure_hindi(s):
                    warning_sentences.append(s)
            if warning_sentences:
                caveat_block = (
                    "शुरू करने से पहले फोटो को लेकर एक छोटी सी बात। "
                    + " ".join(warning_sentences)
                    + " अगली बार अच्छी रोशनी में और कैमरे की ओर सीधा देखकर फोटो लेना मददगार रहेगा।"
                )

        # ── Overall result ────────────────────────────────────────────────────
        label = p4.overall_label
        spoken_label_hi = _HI_SPOKEN_LABEL.get(label, label)

        opening = f"{greeting} {now_hi}. कुल मिलाकर, {spoken_label_hi}."
        conf_note_hi = _observation_score_caveat_hi(p4.observation_score)
        if conf_note_hi:
            opening += f" {conf_note_hi}"
        blocks.append(opening)

        if caveat_block:
            blocks.append(caveat_block)

        if p4.overall:
            blocks.append(_join_findings_hi(p4.overall))

        # ── Face balance ──────────────────────────────────────────────────────
        face_lines = [t for t in [
            p4.face.symmetry, p4.face.alignment,
            p4.face.proportion, p4.face.swelling
        ] if t]
        if face_lines:
            blocks.append(_section_hi(
                "आपके चेहरे के आकार और संतुलन की बात करें तो,", _join_findings_hi(face_lines)
            ))

        # ── Eyes ──────────────────────────────────────────────────────────────
        eye_obs = p4.eyes.left + p4.eyes.right
        non_normal_eyes = [o for o in eye_obs if not o.is_normal]
        if not non_normal_eyes:
            blocks.append("आपकी आंखें स्वस्थ दिख रही हैं, थकान, सूजन या लालिमा का कोई संकेत नहीं है।")
        else:
            blocks.append(_section_hi(
                "आपकी आंखों की बात करें तो,",
                _join_findings_hi([o.finding for o in non_normal_eyes]),
            ))

        if p4.eyes.openness_note:
            cleaned_note = _period(translate_finding(_clean(p4.eyes.openness_note, capitalize=False)))
            if cleaned_note and is_pure_hindi(cleaned_note):
                blocks.append(cleaned_note)

        # ── Skin ──────────────────────────────────────────────────────────────
        skin_obs = (
            p4.skin.forehead + p4.skin.left_cheek +
            p4.skin.right_cheek + p4.skin.nose + p4.skin.overall
        )
        non_normal_skin = [o for o in skin_obs if not o.is_normal]
        if not non_normal_skin:
            blocks.append("आपकी त्वचा साफ़ दिख रही है, कोई खास मुंहासे, लालिमा या असमान बनावट नहीं है।")
        else:
            blocks.append(_section_hi(
                "आपकी त्वचा की बात करें तो,",
                _join_findings_hi([o.finding for o in non_normal_skin]),
            ))

        # ── Lips ──────────────────────────────────────────────────────────────
        if not p3.lips.available:
            blocks.append("इस तस्वीर से होंठों के बारे में ठीक से पता नहीं चल पाया।")
        else:
            non_normal_lips = [o for o in p4.lips.findings if not o.is_normal]
            if not non_normal_lips:
                blocks.append("आपके होंठ स्वस्थ दिख रहे हैं, सूखापन या पीलापन का कोई संकेत नहीं है।")
            else:
                blocks.append(_section_hi(
                    "आपके होंठों की बात करें तो,",
                    _join_findings_hi([o.finding for o in non_normal_lips]),
                ))

        # ── Mouth ─────────────────────────────────────────────────────────────
        if p3.smile.available:
            non_normal_mouth = [o for o in p4.mouth.findings if not o.is_normal]
            if not non_normal_mouth:
                blocks.append("आपके मुंह का आकार संतुलित और सम है।")
            else:
                blocks.append(_section_hi(
                    "आपके मुंह के आकार की बात करें तो,",
                    _join_findings_hi([o.finding for o in non_normal_mouth]),
                ))

        # ── Closing ───────────────────────────────────────────────────────────
        all_obs = (
            p4.eyes.left + p4.eyes.right +
            p4.skin.forehead + p4.skin.left_cheek +
            p4.skin.right_cheek + p4.skin.nose + p4.skin.overall +
            p4.lips.findings + p4.mouth.findings
        )
        non_normal = [o for o in all_obs if not o.is_normal]
        notable    = [o for o in non_normal
                      if o.severity.value.lower() in ("moderate", "notable")]

        sign_off = f", {name}" if name else ""
        outro = f"यह रहा आपका पूरा सारांश{sign_off}। "

        if not non_normal:
            closing = (
                outro + "सब कुछ ठीक दिख रहा है, ऐसे ही बने रहें। "
                "अपनी प्रगति देखते रहने के लिए कुछ दिनों में फिर से जांच करें।"
            )
        elif notable:
            closing = (
                outro + "कुछ बातों की जांच किसी विशेषज्ञ से करवाना अच्छा रहेगा, "
                "पर इन्हें इस तरह जल्दी पकड़ना एक अच्छा कदम है। "
                "यह कोई निदान नहीं है, बस अपना ख्याल रखने की एक सलाह भर है।"
            )
        else:
            closing = (
                outro + "कुछ छोटी बातें सामने आई हैं, पर घबराने की कोई बात नहीं है। "
                "आमतौर पर आराम और पानी पीने से ये ठीक हो जाती हैं। "
                "कुछ दिनों में दोबारा जांच करके देखें कि हालात कैसे बदलते हैं।"
            )
        blocks.append(closing)

        return "\n\n".join(b for b in blocks if b) + "\n"