# PhishGuard AI: Hybrid Phishing Detection System

## Project Overview
This project was developed for the **CI601 University of Brighton Computing Project**. It implements a hybrid machine learning approach to detect phishing URLs by combining structural lexical features with character-level TF-IDF n-grams.

### Comparative Performance Results
| Metric | Baseline (Numeric) | Hybrid (Numeric + N-grams) | Improvement |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 74.37% | **90.80%** | **+16.43%** |
| **ROC–AUC** | 0.8015 | **0.9647** | **+0.1632** |
| **Recall (Phishing)** | 77.76% | **90.42%** | **+12.66%** |

---

## System Architecture
* **Training Environment**: Python scripts in `/src/` handle cleaning and training.
* **Serving Layer**: A **FastAPI** backend (`app.py`) that loads serialised `.joblib` artefacts.
* **Interaction Layer**: A web dashboard (`/static/`) with human-readable "Analyst Notes."

---

## Installation & Setup
### 1. Environment Setup (Python 3.13)
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Running the System
To start the AI backend, execute:

```Bash
python app.py
```
Once the terminal displays "System is operational," open static/index.html in your browser.

---

### Project Structure
* **/data/:** Processed datasets.

* **/models/:** Serialised model artefacts `(.joblib).`

* **/src/:** Core Python logic and training scripts.

* **/static/:** Frontend web assets (HTML, CSS, JS).

```text
PhishGuardAI/
├── data/                       # Dataset storage
│   ├── processed_features.csv   # Feature-engineered dataset
│   └── raw_data.csv            # Original training data
├── models/                     # Serialised AI Artifacts
│   ├── hybrid_model.joblib      # The trained classifier
│   ├── tfidf_vectorizer.joblib  # Text-to-numeric vectorizer
│   ├── scaler.joblib            # Feature scaling parameters
│   └── feature_names.joblib     # Ordered list of training features
├── src/                        # Source Code
│   ├── __init__.py             # Makes directory a Python package
│   ├── feature_engineering.py   # Core URL parsing & entropy logic
│   └── model_training.py       # Script used to generate .joblib files
├── static/                     # Frontend Assets
│   └── index.html              # The Dashboard UI & JavaScript logic
├── venv/                       # Virtual Environment (Local only)
├── app.py                      # FastAPI Backend Entry Point
├── requirements.txt            # Dependency list (FastAPI, Scikit-learn, etc.)
├── .gitignore                  # Instructions to exclude venv/ from Git
└── README.md                   # Project documentation & setup guide
```

---

### Developer Notes
I capped the TF-IDF features at 5,000 to maintain low-latency (under 150ms).

**Security Note:** This prototype uses an academic dataset and is not connected to a live web crawler for safety during the CI601 assessment.
