"""
Phase 6 -- Hindi Translation Toggle (TTS / report output text)
----------------------------------------------------------------
Rule-based English -> Hindi (Devanagari) translation applied to the
generated TTS report text, selected by passing language="hi" to
TTSReportGenerator.generate().

Why rule-based instead of a generic translator:
    tts_report.py builds its output from a small, FIXED set of scaffold
    sentences (greeting, section intros, closings, the spoken_label map,
    the observation-score caveat) plus interpolated "finding" sentences
    that ultimately come from a finite template catalog in
    phase4_observation_engine/observation_engine.py, further normalized
    by tts_report.py's own _REPLACEMENTS list into a constrained,
    plain-language vocabulary. That means the text is NOT free-form --
    it's closer to an i18n message catalog than to arbitrary prose, so a
    two-layer dictionary approach gets most of it right without needing
    network access to an MT API (this container's network is restricted
    to package registries, not general internet/translation services).

Layers, applied in order:
    1. SCAFFOLD_PHRASES -- exact/near-exact whole-sentence and
       whole-clause translations for the fixed wrapper text (greeting,
       section intros, closings, spoken_label, quality caveat, the
       observation-score caveat, the medical-disclaimer boilerplate).
    2. CONNECTORS -- the "Also,"/"On top of that,"/"As well," linking
       words tts_report.py uses to string multiple findings together,
       swapped for natural spoken Hindi connectors.
    3. DATE_PATTERN -- "27 September 2026" style dates: only the month
       name is localized (digits stay Latin, which is completely normal
       in spoken/written modern Hindi -- nobody says "सत्ताईस").
    4. TEMPLATE_PATTERNS -- structural regexes for the actual
       interpolated finding sentences (skin/lip/mouth findings, the
       face-shape summary, the "N observation(s) detected" summary).
       These re-order the sentence into correct Hindi word order
       (subject/region first, "पर", then the finding) instead of
       swapping words in place inside English syntax -- which is what
       made the old output read as broken Hinglish ("mild त्वचा त्वचा
       की असमान बनावट on माथा" instead of "माथे पर हल्की त्वचा में
       हल्की असमानता"). This layer also fixes the literal source of the
       repeated-त्वचा bug: the English template says "{severity} skin
       {finding} on {region}", and when {finding} already contains the
       word "skin" (e.g. "uneven skin texture"), naive word-substitution
       translated the leading category word "skin" AND the finding
       phrase separately, producing "skin" twice. The template patterns
       consume that leading category word without re-translating it
       (the section intro, e.g. "आपकी त्वचा की बात करें तो,", already
       established we're talking about skin), so it only appears once.
    5. WORD_PHRASES -- a word/phrase substitution table for whatever
       vocabulary is left over (used both as a generic fallback and as
       the lookup table the template patterns above call into for the
       actual finding word, e.g. "redness" -> "लालिमा").
    6. A light cleanup pass that collapses an accidental immediate
       repeat of a small set of known nouns (a residual safety net for
       any finding shape not covered by layer 4).

Known limitation: layer 5 is still substitution, not grammatical
translation, so a genuinely novel finding sentence not covered by
either the template patterns or the word dictionary may still come out
as a Hindi/English mix rather than fluent Hindi. That's a graceful
degradation (nothing is dropped or garbled into nonsense). If you hit
more of these, the fastest way to close the gap for good is to share
tts_report.py and observation_engine.py's finding-template catalog so
every possible sentence shape can get its own TEMPLATE_PATTERNS entry
instead of falling back to word-by-word substitution. The intended
longer-term follow-up is still to route uncovered text through the
project's existing Ollama pipeline (already used for OllamaVoice prompt
tuning) once that's wired up for translation specifically.
"""

import re

# ── Layer 1: fixed scaffold sentences/clauses (hand-translated) ────────────
# Ordered roughly by how they appear in tts_report.py's _build().
SCAFFOLD_PHRASES = [
    # Greeting
    ("Hi {name}, here is your facial health summary for",
     "नमस्ते {name}, यह रहा आपका चेहरे की सेहत का सारांश"),
    ("Here is your facial health summary for",
     "यह रहा आपका चेहरे की सेहत का सारांश"),
    ("Overall,", "कुल मिलाकर,"),

    # spoken_label map (overall result)
    ("you're doing great, nothing concerning stood out",
     "आप बहुत अच्छे हैं, कोई चिंता की बात नहीं दिखी"),
    ("you're doing well overall, with just a couple of small things worth a little attention",
     "आप कुल मिलाकर ठीक हैं, बस एक-दो छोटी बातों पर थोड़ा ध्यान देने की ज़रूरत है"),
    ("there are a few things worth keeping an eye on, though nothing here should worry you",
     "कुछ बातों पर नज़र रखने लायक हैं, पर इनमें से किसी से घबराने की ज़रूरत नहीं है"),
    ("a few things stood out that are worth getting checked, and catching them early is a good move",
     "कुछ बातें सामने आई हैं जिन्हें जांच करवाना अच्छा रहेगा, और इन्हें जल्दी पकड़ना एक अच्छा कदम है"),

    # Observation-score caveat
    ("This reading has a moderate observation score, so treat it as a helpful first look.",
     "इस बार फोटो से पूरी तरह साफ़-साफ़ जानकारी नहीं मिल पाई, तो इसे बस एक हल्की झलक जैसा ही समझें।"),
    ("This reading has a lower observation score, so it's worth retaking the photo for a clearer picture.",
     "इस बार फोटो से ठीक से पहचान नहीं हो पाई, तो अच्छी रोशनी में एक और फोटो लेना बेहतर रहेगा।"),

    # Quality caveat
    ("Before we begin, a quick note on photo quality.",
     "शुरू करने से पहले फोटो को लेकर एक छोटी सी बात।"),
    ("A retake in good lighting, facing the camera, would help next time.",
     "अगली बार अच्छी रोशनी में और कैमरे की ओर सीधा देखकर फोटो लेना मददगार रहेगा।"),

    # Section intros
    ("For your face shape and balance,", "आपके चेहरे के आकार और संतुलन की बात करें तो,"),
    ("For your eyes,", "आपकी आंखों की बात करें तो,"),
    ("Your eyes look healthy, no signs of tiredness, puffiness, or redness.",
     "आपकी आंखें स्वस्थ दिख रही हैं, थकान, सूजन या लालिमा का कोई संकेत नहीं है।"),
    ("On your skin,", "आपकी त्वचा की बात करें तो,"),
    ("Your skin looks clear, no significant breakouts, redness, or uneven texture.",
     "आपकी त्वचा साफ़ दिख रही है, कोई खास मुंहासे, लालिमा या असमान बनावट नहीं है।"),
    ("Lip analysis wasn't available from this image.",
     "इस तस्वीर से होंठों के बारे में ठीक से पता नहीं चल पाया।"),
    ("For your lips,", "आपके होंठों की बात करें तो,"),
    ("Your lips look healthy, no signs of dryness or paleness.",
     "आपके होंठ स्वस्थ दिख रहे हैं, सूखापन या पीलापन का कोई संकेत नहीं है।"),
    ("For your mouth shape,", "आपके मुंह के आकार की बात करें तो,"),
    ("Your mouth shape looks balanced and even.",
     "आपके मुंह का आकार संतुलित और सम है।"),

    # Medical disclaimer boilerplate (seen appended to the overall-observations line)
    ("this report contains visual observations only and does not constitute a medical diagnosis.",
     "यह जानकारी सिर्फ देखने में मिली बातों पर आधारित है, यह किसी बीमारी की पक्की पहचान नहीं है।"),
    ("please consult a qualified healthcare professional for any health concerns.",
     "सेहत से जुड़ी किसी भी चिंता के लिए कृपया किसी डॉक्टर से सलाह लें।"),

    # Trailing clauses on specific findings
    ("possible indicator of reduced circulation in lip area.",
     "जो खून की रवानी में कमी का संकेत भी हो सकता है।"),
    ("may reflect natural asymmetry or positional artifact.",
     "यह स्वाभाविक बनावट की वजह से भी हो सकता है, या फिर फोटो खिंचवाते वक्त चेहरे की स्थिति की वजह से भी।"),

    # Face-shape summary sentences (fixed shape, only inserted when true)
    ("face symmetry appears normal.",
     "चेहरे की बनावट सामान्य और संतुलित लग रही है।"),
    ("head alignment appears level.",
     "सिर की सीध भी ठीक लग रही है।"),
    ("facial proportions appear within typical range.",
     "चेहरे का अनुपात सामान्य दायरे में है।"),
    ("no swelling indicators detected based on facial proportions.",
     "चेहरे की बनावट को देखते हुए सूजन का कोई संकेत नहीं मिला।"),

    # Closings (name inserted via {name})
    ("That wraps up your summary{sign_off}. Everything looks good, keep it up. "
     "Check back in a few days to keep tracking your progress.",
     "यह रहा आपका पूरा सारांश{sign_off}। सब कुछ ठीक दिख रहा है, ऐसे ही बने रहें। "
     "अपनी प्रगति देखते रहने के लिए कुछ दिनों में फिर से जांच करें।"),
    ("That wraps up your summary{sign_off}. A few things are worth getting checked by a "
     "professional, but catching them early like this is a good move. "
     "This isn't a diagnosis, just a helpful nudge to look after yourself.",
     "यह रहा आपका पूरा सारांश{sign_off}। कुछ बातों की जांच किसी विशेषज्ञ से करवाना अच्छा रहेगा, "
     "पर इन्हें इस तरह जल्दी पकड़ना एक अच्छा कदम है। "
     "यह कोई निदान नहीं है, बस अपना ख्याल रखने की एक सलाह भर है।"),
    ("That wraps up your summary{sign_off}. A few small things showed up, but nothing "
     "to worry about. These often ease up with rest and hydration. "
     "Try again in a few days to see how things are tracking.",
     "यह रहा आपका पूरा सारांश{sign_off}। कुछ छोटी बातें सामने आई हैं, पर घबराने की कोई बात नहीं है। "
     "आमतौर पर आराम और पानी पीने से ये ठीक हो जाती हैं। "
     "कुछ दिनों में दोबारा जांच करके देखें कि हालात कैसे बदलते हैं।"),
]

# ── Layer 2: the "Also," / "On top of that," / "As well," linking words
# tts_report.py uses to string several findings from the same section
# together. Kept lowercase; matched case-insensitively (see _compile).
CONNECTORS = [
    ("also,", "साथ ही,"),
    ("on top of that,", "इसके अलावा,"),
    ("as well,", "और,"),
]

# ── Layer 3: dates -- only the month name is localized, digits stay as-is.
_MONTHS_HI = {
    "january": "जनवरी", "february": "फ़रवरी", "march": "मार्च",
    "april": "अप्रैल", "may": "मई", "june": "जून", "july": "जुलाई",
    "august": "अगस्त", "september": "सितंबर", "october": "अक्टूबर",
    "november": "नवंबर", "december": "दिसंबर",
}
_DATE_RE = re.compile(
    r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+(\d{4})\b",
    re.IGNORECASE,
)


def _translate_dates(text: str) -> str:
    def repl(match):
        day, month, year = match.group(1), match.group(2), match.group(3)
        return f"{day} {_MONTHS_HI.get(month.lower(), month)} {year}"

    return _DATE_RE.sub(repl, text)


# ── Severity vocabulary, shared by every TEMPLATE_PATTERNS builder below.
# Deliberately conversational rather than literary/Sanskritized -- this is
# meant to sound like how the finding would actually get said out loud.
# Gendered so it can agree with whatever Hindi noun it ends up next to
# ("हल्की असमानता" vs "हल्का अंतर") -- "inv" is used for adjective phrases
# that don't inflect by gender either way (ज़्यादा/लायक-based ones).
_SEVERITY_HI = {
    "slight": {"m": "हल्का", "f": "हल्की", "inv": "हल्का"},
    "mild": {"m": "हल्का", "f": "हल्की", "inv": "हल्का"},
    "moderate": {"m": "थोड़ा ज़्यादा", "f": "थोड़ी ज़्यादा", "inv": "थोड़ा ज़्यादा"},
    "noticeable": {"m": "ध्यान देने योग्य", "f": "ध्यान देने योग्य", "inv": "ध्यान देने योग्य"},
    "notable": {"m": "ध्यान देने योग्य", "f": "ध्यान देने योग्य", "inv": "ध्यान देने योग्य"},
    "significant": {"m": "काफी ज़्यादा", "f": "काफी ज़्यादा", "inv": "काफी ज़्यादा"},
}


def _severity_hi(word: str, gender: str = "inv") -> str:
    entry = _SEVERITY_HI.get(word.lower())
    if not entry:
        return word
    return entry.get(gender, entry["inv"])


# Oblique forms used with "पर" ("on the forehead" -> "माथे पर").
_REGION_ON_HI = {
    "forehead": "माथे",
    "left cheek": "बाएं गाल",
    "right cheek": "दाएं गाल",
    "nose": "नाक",
    "chin": "ठुड्डी",
    "jaw": "जबड़े",
    "neck": "गर्दन",
}

# Known skin-finding phrases get a hand-written Hindi clause (with the
# right grammatical gender for the severity word to agree with) instead
# of plain word-substitution, since "{severity} {finding}" word-swapped
# in place reads as broken Hinglish rather than a real sentence.
_SKIN_FINDING_META = {
    "uneven skin texture": ("त्वचा की बनावट में {sev} असमानता", "f"),
    "redness": ("{sev} लालिमा", "f"),
    "colour evenness": ("त्वचा के रंग में {sev} अंतर", "m"),
    "color evenness": ("त्वचा के रंग में {sev} अंतर", "m"),
}


# ── Layer 5 dictionary (declared here, used by both layer 4 and layer 5) ──
# Longest phrases first so multi-word terms match before their sub-words do.
WORD_PHRASES = [
    ("uneven skin texture", "त्वचा की असमान बनावट"),
    ("dark area under the eye", "आंख के नीचे का काला क्षेत्र"),
    ("how open the eye appears", "आंख के खुलेपन की स्थिति"),
    ("puffiness under the eye", "आंख के नीचे की सूजन"),
    ("reduced color saturation", "रंगत में कमी"),
    ("reduced colour saturation", "रंगत में कमी"),
    ("colour evenness", "रंग की समरूपता"),
    ("color evenness", "रंग की समरूपता"),
    ("mouth curvature", "मुंह की वक्रता"),
    ("face shape", "चेहरे का आकार"),
    ("face balance", "चेहरे का संतुलन"),
    ("head tilt", "सिर का झुकाव"),
    ("breakout level", "मुंहासों का स्तर"),
    ("circulation", "खून की रवानी"),
    ("indicators", "संकेत"),
    ("indicator", "संकेत"),
    ("significant", "काफी ज़्यादा"),
    ("noticeable", "ध्यान देने योग्य"),
    ("notable", "ध्यान देने योग्य"),
    ("moderate", "थोड़ा ज़्यादा"),
    ("slight", "हल्का"),
    ("mild", "हल्का"),
    ("reduced", "कम"),
    ("paleness", "पीलापन"),
    ("redness", "लालिमा"),
    ("unevenness", "असमानता"),
    ("puffiness", "सूजन"),
    ("dryness", "सूखापन"),
    ("healthy", "स्वस्थ"),
    ("normal", "सामान्य"),
    ("left eye", "बाईं आंख"),
    ("right eye", "दाईं आंख"),
    ("left cheek", "बाएं गाल"),
    ("right cheek", "दाएं गाल"),
    ("forehead", "माथा"),
    ("nose", "नाक"),
    ("skin", "त्वचा"),
    ("lips", "होंठ"),
    ("mouth", "मुंह"),
    ("eyes", "आंखें"),
    ("face", "चेहरा"),
]

_SCAFFOLD_COMPILED = None
_CONNECTORS_COMPILED = None
_WORD_COMPILED = None


def _compile(table, ignorecase=False):
    """Build one alternation regex, longest keys first."""
    ordered = sorted(table, key=lambda kv: len(kv[0]), reverse=True)
    mapping = dict(ordered)
    pattern = "|".join(re.escape(key) for key, _ in ordered)
    flags = re.IGNORECASE if ignorecase else 0
    return re.compile(pattern, flags), mapping


def _apply(table_regex_map, text):
    regex, mapping = table_regex_map
    if not text:
        return text

    def repl(match):
        matched = match.group(0)
        # Exact-case hit first, then a case-insensitive fallback (covers
        # a dictionary word like "redness" appearing capitalized at the
        # start of a sentence, e.g. "Redness was noticeable...").
        return mapping.get(matched, mapping.get(matched.lower(), matched))

    return regex.sub(repl, text)


def _word_compiled():
    global _WORD_COMPILED
    if _WORD_COMPILED is None:
        _WORD_COMPILED = _compile(WORD_PHRASES, ignorecase=True)
    return _WORD_COMPILED


def _translate_words(text: str) -> str:
    return _apply(_word_compiled(), text)


# ── Layer 4: structural patterns for the interpolated finding sentences.
# Each rewrites English word order into Hindi word order instead of
# swapping words in place -- this is what actually fixes the "mild
# त्वचा त्वचा की असमान बनावट on माथा" style output.

_SKIN_FINDING_RE = re.compile(
    r"\b(mild|moderate|significant|notable|noticeable|slight)\s+skin\s+"
    r"(.+?)\s+on\s+([a-zA-Z]+(?:\s[a-zA-Z]+)?)(?=[.,]|$)",
    re.IGNORECASE,
)


def _skin_finding_repl(match):
    severity, finding, region = match.group(1), match.group(2), match.group(3)
    region_hi = _REGION_ON_HI.get(region.strip().lower(), region)
    # NOT prefixed with a separately-translated "skin" -- so a finding
    # that already says "uneven skin texture" doesn't end up with
    # त्वचा twice. The section intro ("आपकी त्वचा की बात करें तो,")
    # already tells the listener we're talking about skin.
    meta = _SKIN_FINDING_META.get(finding.strip().lower())
    if meta:
        template, gender = meta
        finding_hi = template.format(sev=_severity_hi(severity, gender))
    else:
        # Unknown finding phrase -- graceful degradation via plain
        # word-substitution rather than a hand-written clause.
        finding_hi = f"{_severity_hi(severity)} {_translate_words(finding)}"
    return f"{region_hi} पर {finding_hi} है।"


_LIP_FINDING_RE = re.compile(
    r"\b(mild|moderate|significant|notable|noticeable|slight)\s+lip\s+"
    r"(dryness|paleness)(?:\s*\(([^)]*)\))?\s+observed\.?",
    re.IGNORECASE,
)


def _lip_finding_repl(match):
    severity, condition, paren = match.group(1), match.group(2), match.group(3)
    sev_hi = _severity_hi(severity)
    cond_hi = _translate_words(condition)
    if paren:
        paren_hi = _translate_words(paren)
        return f"होंठों में {sev_hi} {cond_hi} है ({paren_hi})।"
    return f"होंठों में {sev_hi} {cond_hi} है।"


_MOUTH_CORNER_RE = re.compile(
    r"\b(mild|moderate|significant|notable|noticeable|slight)\s+difference\s+in\s+"
    r"mouth\s+corner\s+height\s+(?:noted\s+)?between\s+left\s+and\s+right\.?",
    re.IGNORECASE,
)

_EYE_OPENNESS_RE = re.compile(
    r"\b(mild|moderate|significant|notable|noticeable|slight)\s+difference\s+in\s+"
    r"eye\s+openness\s+(?:noted|between\s+left\s+and\s+right)\.?",
    re.IGNORECASE,
)


def _eye_openness_repl(match):
    severity = match.group(1)
    sev_hi = _severity_hi(severity, "m")  # अंतर is masculine
    return f"आंखों के खुलेपन में {sev_hi} अंतर देखा गया।"


_FACE_ASYMMETRY_RE = re.compile(
    r"\b(mild|moderate|significant|notable|noticeable|slight)\s+facial\s+"
    r"(?:asymmetry|unevenness)\s+noted\.?",
    re.IGNORECASE,
)


def _face_asymmetry_repl(match):
    severity = match.group(1)
    sev_hi = _severity_hi(severity, "f")  # विषमता is feminine
    return f"चेहरे में {sev_hi} विषमता देखी गई।"


_HEAD_TILT_RE = re.compile(
    r"\b(mild|moderate|significant|notable|noticeable|slight)\s+head\s+tilt\s+observed\.?",
    re.IGNORECASE,
)


def _head_tilt_repl(match):
    severity = match.group(1)
    sev_hi = _severity_hi(severity, "m")  # झुकाव is masculine
    return f"सिर में {sev_hi} झुकाव देखा गया।"


# Eye findings shaped like skin findings ("{severity} {finding} on {left/right} eye")
# but without the "skin" keyword, so _SKIN_FINDING_RE doesn't match them.
_EYE_REGION_HI = {"left eye": "बाईं आंख", "right eye": "दाईं आंख"}

_EYE_FINDING_META = {
    "puffiness under the eye": ("आंख के नीचे {sev} सूजन", "f"),
    "dark area under the eye": ("आंख के नीचे {sev} कालापन", "m"),
    "puffiness": ("आंख के आसपास {sev} सूजन", "f"),
}

_EYE_FINDING_RE = re.compile(
    r"\b(mild|moderate|significant|notable|noticeable|slight)\s+"
    r"(.+?)\s+on\s+(left eye|right eye)\b\.?",
    re.IGNORECASE,
)


def _eye_finding_repl(match):
    severity, finding, eye = match.group(1), match.group(2), match.group(3)
    eye_hi = _EYE_REGION_HI.get(eye.strip().lower(), eye)
    meta = _EYE_FINDING_META.get(finding.strip().lower())
    if meta:
        template, gender = meta
        finding_hi = template.format(sev=_severity_hi(severity, gender))
    else:
        finding_hi = f"{_severity_hi(severity)} {_translate_words(finding)}"
    return f"{eye_hi} में {finding_hi} है।"


def _mouth_corner_repl(match):
    severity = match.group(1)
    sev_hi = _severity_hi(severity, "m")  # फ़र्क़ is masculine
    return f"मुंह के दोनों कोनों की ऊंचाई में {sev_hi} फ़र्क़ है।"


_OBSERVATION_COUNT_RE = re.compile(
    r"\b(mild|moderate|significant|notable|noticeable|slight)\s+"
    r"observation\(s\)\s+(detected requiring attention|identified|noted)\.?",
    re.IGNORECASE,
)

_OBSERVATION_VERB_HI = {
    "detected requiring attention": "मिली हैं जिन पर ध्यान देना ज़रूरी है",
    "identified": "भी मिली हैं",
    "noted": "भी दर्ज़ हुई हैं",
}


def _observation_count_repl(match):
    severity, verb = match.group(1), match.group(2)
    sev_hi = _severity_hi(severity)
    verb_hi = _OBSERVATION_VERB_HI.get(verb.lower(), verb)
    return f"{sev_hi} स्तर की कुछ बातें {verb_hi}।"


_TEMPLATE_PATTERNS = [
    (_OBSERVATION_COUNT_RE, _observation_count_repl),
    (_SKIN_FINDING_RE, _skin_finding_repl),
    (_LIP_FINDING_RE, _lip_finding_repl),
    (_MOUTH_CORNER_RE, _mouth_corner_repl),
    (_EYE_OPENNESS_RE, _eye_openness_repl),
    (_FACE_ASYMMETRY_RE, _face_asymmetry_repl),
    (_HEAD_TILT_RE, _head_tilt_repl),
    (_EYE_FINDING_RE, _eye_finding_repl),
]

# ── Fixed-shape findings that come back verbatim from the observation
# engine for the "nothing wrong here" case (see the matching English
# literals in SCAFFOLD_PHRASES's face-shape block above). Exact,
# case-insensitive match against the whole (already-cleaned) finding text
# -- checked before the structural templates, since these aren't
# region+severity shaped like a skin/lip finding.
FINDING_SCAFFOLD = {
    "face symmetry appears normal.":
        "चेहरे की बनावट सामान्य और संतुलित लग रही है।",
    "head alignment appears level.":
        "सिर की सीध भी ठीक लग रही है।",
    "facial proportions appear within typical range.":
        "चेहरे का अनुपात सामान्य दायरे में है।",
    "no swelling indicators detected based on facial proportions.":
        "चेहरे की बनावट को देखते हुए सूजन का कोई संकेत नहीं मिला।",
    "face appears relatively wide for its height.":
        "चेहरा अपनी ऊंचाई के हिसाब से थोड़ा चौड़ा लग रहा है।",
}

# Long, specific trailing clauses -- safe to substring-match (unlike short
# severity words) because they're long enough that they won't accidentally
# appear inside an unrelated word.
_LONG_CLAUSES = [
    ("possible indicator of reduced circulation in lip area.",
     "जो खून की रवानी में कमी का संकेत भी हो सकता है।"),
    ("may reflect natural asymmetry or positional artifact.",
     "यह स्वाभाविक बनावट की वजह से भी हो सकता है, या फिर फोटो खिंचवाते वक्त चेहरे की स्थिति की वजह से भी।"),
    ("minor variations are common and typically insignificant.",
     "थोड़े-बहुत अंतर सामान्य होते हैं और ज़्यादातर मामूली होते हैं।"),
    ("may be positional.",
     "यह सिर की स्थिति की वजह से भी हो सकता है।"),
    ("may indicate asymmetric muscle tone or positional artifact.",
     "यह मांसपेशियों की स्थिति या फोटो खिंचवाते समय चेहरे की स्थिति की वजह से भी हो सकता है।"),
    ("this report contains visual observations only and does not constitute a medical diagnosis.",
     "यह जानकारी सिर्फ देखने में मिली बातों पर आधारित है, यह किसी बीमारी की पक्की पहचान नहीं है।"),
    ("please consult a qualified healthcare professional for any health concerns.",
     "सेहत से जुड़ी किसी भी चिंता के लिए कृपया किसी डॉक्टर से सलाह लें।"),
]


def _apply_templates(text: str) -> str:
    for regex, repl in _TEMPLATE_PATTERNS:
        text = regex.sub(repl, text)
    return text


# ── Layer 6: collapse an accidental immediate repeat of a known noun.
# Scoped to a fixed word list (rather than any \1 \1 pattern) so genuine
# Hindi reduplication -- "धीरे-धीरे", "जल्दी-जल्दी" -- is never touched;
# this only cleans up the kind of artifact layer 4 is designed to avoid
# in the first place, as a safety net for any finding shape it doesn't
# recognize yet.
_DEDUPE_NOUNS = {"त्वचा", "आंख", "आंखें", "गाल", "होंठ", "मुंह", "चेहरा", "माथा", "नाक"}
_DEDUPE_RE = re.compile(r"\b(\S+)\s+\1\b")


def _dedupe_known_repeats(text: str) -> str:
    def repl(match):
        word = match.group(1)
        return word if word in _DEDUPE_NOUNS else match.group(0)

    return _DEDUPE_RE.sub(repl, text)


_LATIN_RE = re.compile(r"[A-Za-z]")


def is_pure_hindi(text: str) -> bool:
    """True if text has no Latin-alphabet characters left in it.
    A Hindi TTS voice can't read Latin script -- a leftover English word or
    fragment doesn't just sound wrong, it tends to come out as dead air or
    a dropped segment. Callers use this to keep any not-yet-covered
    finding out of the Hindi script entirely rather than emitting a
    mixed-language sentence."""
    return not _LATIN_RE.search(text)


def translate_finding(text: str) -> str:
    """
    Translate ONE already-cleaned finding sentence (a single region/skin/
    lip/mouth/eye finding, or a single quality-warning line) to Hindi.

    This is deliberately scoped to a single short finding -- never the
    fully assembled, multi-sentence report -- because running the
    template/word layers over a whole paragraph lets a dictionary entry
    match a substring buried inside an unrelated word (e.g. "significant"
    matching inside "insignificant"), which is how earlier report runs
    ended up with glued-together garbage like "typically inकाफी ज़्यादा".
    Called per-finding, before findings are joined with Hindi connectors,
    each regex only ever sees a small, well-shaped string, which removes
    that failure mode. Scaffold sentences (greetings, section intros,
    closings) are not handled here -- callers should use native Hindi
    scaffold text for those rather than routing them through this
    function.
    """
    if not text:
        return text

    # Exact-match fixed findings (the "nothing notable here" face-shape
    # sentences) -- checked whole-string first since they aren't
    # region+severity shaped.
    exact = FINDING_SCAFFOLD.get(text.strip().lower())
    if exact:
        return exact

    result = text
    result = _translate_dates(result)
    result = _apply_templates(result)
    for en, hi in _LONG_CLAUSES:
        result = re.sub(re.escape(en), hi, result, flags=re.IGNORECASE)
    result = _translate_words(result)
    result = _dedupe_known_repeats(result)
    return result


def translate_tts_text(text: str) -> str:
    """
    Translate a fully-built English TTS report string to Hindi.

    Placeholders like {name} and {sign_off} in SCAFFOLD_PHRASES are
    matched against the ALREADY interpolated English text, so this only
    rewrites the literal boilerplate; any {name}/{sign_off} placeholder
    syntax left in this module's own strings is for readability of the
    table only and is never applied to already-rendered report text
    (those calls already format() their own name/sign_off before this
    function ever sees them -- see the two-pass handling below).
    """
    global _SCAFFOLD_COMPILED, _CONNECTORS_COMPILED
    if not text:
        return text

    result = text

    # 1a. Direct (parameter-free) scaffold phrases.
    direct = [(en, hi) for en, hi in SCAFFOLD_PHRASES if "{" not in en]
    if _SCAFFOLD_COMPILED is None:
        _SCAFFOLD_COMPILED = _compile(direct, ignorecase=True)
    result = _apply(_SCAFFOLD_COMPILED, result)

    # 1b. Closing templates: strip the {sign_off} slot out and match the
    # two halves, preserving whatever sign_off text (", Name" or "") is
    # actually present.
    for en_template, hi_template in SCAFFOLD_PHRASES:
        if "{sign_off}" not in en_template:
            continue
        en_prefix, en_suffix = en_template.split("{sign_off}")
        hi_prefix, hi_suffix = hi_template.split("{sign_off}")
        pattern = re.compile(re.escape(en_prefix) + r"(.*?)" + re.escape(en_suffix), re.DOTALL)
        result = pattern.sub(lambda m: hi_prefix + m.group(1) + hi_suffix, result)

    # 1c. Greeting with {name}: match "Hi <name>, here is..." leaving <name> intact.
    greet_pattern = re.compile(r"Hi (.+?), here is your facial health summary for")
    result = greet_pattern.sub(
        lambda m: f"नमस्ते {m.group(1)}, यह रहा आपका चेहरे की सेहत का सारांश",
        result,
    )

    # 2. Connectors joining multiple findings in the same section.
    if _CONNECTORS_COMPILED is None:
        _CONNECTORS_COMPILED = _compile(CONNECTORS, ignorecase=True)
    result = _apply(_CONNECTORS_COMPILED, result)

    # 3. Dates.
    result = _translate_dates(result)

    # 4. Structural finding templates -- must run before the generic
    # word layer so severity words / "skin" / "observation(s)" are still
    # intact for the regexes to match on.
    result = _apply_templates(result)

    # 5. Word/phrase layer for whatever remains.
    result = _translate_words(result)

    # 6. Cleanup safety net.
    result = _dedupe_known_repeats(result)

    return result