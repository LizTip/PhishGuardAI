import joblib
import pandas as pd
import numpy as np
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.sparse import hstack, csr_matrix
from fastapi.middleware.cors import CORSMiddleware

# Import the extraction logic from my feature_engineering.py
from src.feature_engineering import extract_url_features, clean_url

app = FastAPI(title="PhishGuard AI: Advanced Detection System")

# Enable CORS to permit the local index.html to communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Global Persistence Layer: Loading Model Artifacts
# ---------------------------------------------------------
MODELS_DIR = "models"

print("Loading AI model and preprocessing artifacts...")
try:
    model = joblib.load(os.path.join(MODELS_DIR, "hybrid_model.joblib"))
    tfidf = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.joblib"))
    print("All artifacts successfully loaded. System is operational.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to initialise models. {e}")


class URLInput(BaseModel):
    url: str


def generate_analyst_notes(prediction, prob, features):
    """
    Generates XAI justification.
    """
    if prediction == "Safe" and prob < 0.3:
        return "The URL exhibits structural characteristics consistent with legitimate traffic."

    findings = []
    if features.get('url_entropy', 0) > 4.2:
        findings.append("elevated character entropy")
    if features.get('suspicious_keyword_count', 0) > 0:
        findings.append("sensitive brand-related keywords identified")
    if features.get('subdomain_depth', 0) > 2:
        findings.append("excessive subdomain nesting")
    if features.get('is_https', 1) == 0:
        findings.append("unencrypted HTTP protocol")

    report = f"The classifier flagged this URL as {prediction} with {prob:.1%} confidence."
    if findings:
        report += " Key indicators: " + ", ".join(findings) + "."
    return report


# ---------------------------------------------------------
# Inference Endpoint
# ---------------------------------------------------------
@app.post("/predict")
async def predict(data: URLInput):
    try:
        # 1. URI Sanitisation
        url = clean_url(data.url)

        # 2. Feature Extraction
        lex_features = extract_url_features(url)
        lex_df = pd.DataFrame([lex_features])[feature_names]

        # 3. Parameter Transformation
        lex_scaled = scaler.transform(lex_df)
        tfidf_matrix = tfidf.transform([url])  # Using the loaded 'tfidf' vectorizer

        # 4. Hybrid Matrix Construction
        hybrid_matrix = hstack([lex_scaled, tfidf_matrix])

        # 5. Inference
        prediction_numeric = model.predict(hybrid_matrix)[0]

        # Get probability if model supports it, else use a placeholder
        try:
            prob = model.predict_proba(hybrid_matrix).max()
        except:
            prob = 1.0

        # Map 1 to Phishing and 0 to Safe
        label = "Phishing" if int(prediction_numeric) == 1 else "Safe"

        # 6. Generate XAI Notes
        notes = generate_analyst_notes(label, prob, lex_features)

        # 7. Final Return (This stops the 'null' response!)
        return {
            "prediction": label,
            "probability": float(prob),
            "analyst_notes": notes,
            "url": url
        }

    except Exception as e:
        print(f"Inference Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))