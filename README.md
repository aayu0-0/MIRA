# MIRA

### AI-Powered Facial Health Observation & Tracking System

MIRA is a computer-vision-based digital health observation system designed to identify and track subtle visual changes that are often overlooked in everyday life.

Instead of focusing on a single snapshot, MIRA is designed around **observation, measurement, history, and trends** — helping users understand how small changes may evolve over time.

> **Disclaimer:** MIRA is an educational and research project. It does not diagnose medical conditions and should not be used as a substitute for professional medical advice.

---

## Overview

The human face contains numerous measurable visual features that can be analyzed using computer vision.

MIRA uses facial landmark detection and geometric analysis to extract structured observations from different facial regions. These observations can then be stored and compared over time to identify changes and trends.

The system follows a modular architecture, allowing new facial analysis modules and observation techniques to be added independently.

---

## Core Concept

```text
                    MIRA
                     │
                     ▼
              Input Image
                     │
                     ▼
        Face Detection & Landmarks
                     │
                     ▼
          Facial Region Extraction
                     │
                     ▼
             Feature Analysis
                     │
                     ▼
           Observation Engine
                     │
                     ▼
          History & Trend Analysis
                     │
                     ▼
             Structured Report
```

MIRA focuses on **measurable observations rather than medical diagnosis**.

---

## Features

### Computer Vision

- Facial landmark detection
- Facial region extraction
- Geometric feature analysis
- Eye aspect ratio (EAR)
- Mouth aspect ratio (MAR)
- Facial symmetry analysis
- Region-specific feature extraction

### Observation Engine

- Converts numerical measurements into structured observations
- Applies configurable thresholds
- Generates human-readable observations
- Designed for modular expansion

### Tracking & History

- Stores observations over time
- Maintains historical measurements
- Enables comparison between observations
- Supports trend detection

### Reporting

- Structured observation reports
- Machine-readable data
- Human-readable reports
- Annotated image generation

### Voice Output

MIRA can integrate local text-to-speech using **Piper** for generating spoken feedback from the generated observations.

---

## System Architecture

```text
┌─────────────────────┐
│     Input Image     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Face Understanding  │
│  Detection/Landmark │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Region Extraction   │
│ Eyes / Mouth / Face │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Feature Analysis    │
│ Geometry / Metrics  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Observation Engine  │
│ Thresholds / Rules  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ History & Trends    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Report / Voice      │
└─────────────────────┘
```

---

## Project Structure

```text
MIRA/
│
├── src/
│   │
│   ├── phase1_face_understanding/
│   │   └── Face detection & landmark processing
│   │
│   ├── phase2_region_extraction/
│   │   └── Facial region extraction
│   │
│   ├── phase3_feature_analysis/
│   │   └── Facial feature measurements
│   │
│   ├── phase4_observation_engine/
│   │   └── Observation and scoring logic
│   │
│   ├── tests/
│   │   └── Test suite
│   │
│   ├── pipeline.py
│   └── thresholds.py
│
├── piper/
│   └── Local Piper TTS resources
│
├── data/
│   └── Project data
│
├── Database/
│   └── Historical observation data
│
├── Doc/
│   └── Project documentation
│
├── PHASE_C_AUDIT.md
│
└── README.md
```

---

## Technology Stack

| Technology | Purpose |
|---|---|
| **Python** | Core development |
| **OpenCV** | Image processing |
| **MediaPipe** | Face detection & landmarks |
| **NumPy** | Numerical analysis |
| **SQLite** | Historical data storage |
| **Piper** | Local text-to-speech |
| **Computer Vision** | Facial feature analysis |

---

## Development Pipeline

MIRA is being developed incrementally through independent phases.

### Phase 1 — Face Understanding

Detect the face and obtain reliable facial landmarks.

### Phase 2 — Region Extraction

Organize landmarks into meaningful facial regions such as:

- Eyes
- Mouth
- Face contour
- Other region-specific landmarks

### Phase 3 — Feature Analysis

Convert landmarks into measurable features.

Examples include:

- Eye aspect ratio
- Mouth aspect ratio
- Facial geometry
- Symmetry
- Region-specific measurements

### Phase 4 — Observation Engine

Convert measurements into structured observations using configurable thresholds and rules.

### Phase 5 — History & Trends

Store observations and compare them across multiple sessions to identify changes over time.

### Phase 6 — Reporting & Interaction

Generate structured reports and provide optional voice output using local text-to-speech.

---

## Design Philosophy

MIRA is built around four fundamental principles:

### 1. Measure

Convert visual information into measurable features.

### 2. Observe

Transform measurements into understandable observations.

### 3. Track

Store observations over time rather than relying on a single snapshot.

### 4. Understand

Use historical data to identify patterns and changes.

```text
Measure → Observe → Track → Understand
```

---

## Privacy

MIRA is designed with local processing in mind.

Where possible, analysis and processing can be performed locally instead of sending facial data to external services.

The project is being developed with the goal of keeping sensitive visual data under the user's control.

---

## Current Status

🚧 **Active Development**

Current development areas include:

- [x] Face landmark detection
- [x] Facial region extraction
- [x] Feature analysis
- [x] Observation engine
- [x] Threshold-based scoring
- [x] Historical data architecture
- [x] Report generation
- [x] Local TTS integration
- [ ] Expanded trend analysis
- [ ] Additional observation modules
- [ ] Improved visualization
- [ ] Long-term validation

---

## Installation

Clone the repository:

```bash
git clone https://github.com/aayu0-0/MIRA.git
cd MIRA
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running MIRA

The main pipeline can be executed using:

```bash
python src/pipeline.py
```

> The exact execution flow may change as the project architecture continues to evolve.

---

## Testing

Run the test suite using:

```bash
pytest
```

---

## Future Development

Planned areas of development include:

- More robust facial feature extraction
- Improved calibration
- Additional facial regions
- Longitudinal trend analysis
- Better visualization of historical changes
- More explainable observations
- Improved report generation
- Expanded local AI capabilities
- Privacy-focused local processing
- Integration with additional health-related signals

---

## Important Disclaimer

MIRA is **not a medical diagnostic tool**.

The system produces computer-vision-based observations and measurements for educational and research purposes. These observations should not be interpreted as medical diagnoses, treatment recommendations, or professional medical advice.

If you have concerns about your health, consult a qualified healthcare professional.

---

## Author

**Aayush Sardana**

Instrumentation & Control Engineering  
NIT Jalandhar

GitHub:  
https://github.com/aayu0-0

---

## License

This project is currently under development.

License information will be added as the project reaches a stable release.
