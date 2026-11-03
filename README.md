# EduFeedback AI

An intermediate NLP project for analyzing anonymized college student feedback.

## Features
- Sentiment classification (positive, neutral, negative)
- Keyword-based feedback aspect detection
- Department and sentiment filters
- Charts and downloadable CSV results
- TF-IDF + Logistic Regression training pipeline

## Requirements
- Python 3.10 or newer recommended

## Setup (Windows PowerShell)
```powershell
cd path\to\EduFeedback_AI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
```

If PowerShell blocks activation, use Command Prompt:
```bat
.venv\Scripts\activate.bat
```

## Input CSV format
Required column: `feedback_text`
Optional columns: `department`, `semester`, `aspect`, `feedback_id`

For training, the CSV must also contain `sentiment` with values `Positive`, `Neutral`, or `Negative`. The included sample dataset is illustrative and small; model scores are not reliable evidence of real-world performance.

## Notes
The dashboard can run without the trained model using a simple keyword fallback. Run `train_model.py` to use the machine-learning model. Keyword-based aspect detection is a baseline, not true aspect-based sentiment analysis.

Use anonymized data, secure consent where required, and do not use automated sentiment predictions as the sole basis for decisions about individual students or staff.
