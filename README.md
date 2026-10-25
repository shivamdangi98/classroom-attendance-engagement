# AI-Based Classroom Attendance and Engagement Analysis

A Streamlit application that analyzes classroom video and audio to estimate attendance, engagement patterns, and classroom interaction.

## Features
- Face detection and sustained-presence attendance estimation
- Audio classification: silence, lecture, discussion, and ambient distraction
- Class-level engagement index and visualizations
- Teaching suggestions based on classroom patterns
- CSV attendance export and Markdown/JSON session reports
- Built-in demo mode

## Tech Stack
Python, Streamlit, OpenCV, MediaPipe, NumPy, Pandas, and audio signal processing.

## Run Locally

```bash
python -m venv .venv
```

On Windows:

```cmd
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Privacy and Limitations
Use recordings only with appropriate consent. Engagement and audio classifications are heuristic estimates, not definitive measurements of attention or learning. Do not publish identifiable classroom recordings or student data.

## Disclaimer
Educational demonstration project. Validate outputs before using them for attendance or educational decisions.
