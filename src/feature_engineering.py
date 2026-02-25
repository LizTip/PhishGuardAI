import re
import math
import socket
import warnings
import os
import joblib

import numpy as np
import pandas as pd
import tldextract

from urllib.parse import urlparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Optional imports
try:
    import whois
except Exception:
    whois = None

try:
    import requests
    from bs4 import BeautifulSoup
    from requests.packages.urllib3.exceptions import InsecureRequestWarning
except Exception:
    requests = None
    BeautifulSoup = None
    InsecureRequestWarning = None

# -------------------------
# Config
# -------------------------
DATA_PATH = r"balanced_urls_dataset.csv"

# Network features disabled for stability in prototype
ENABLE_WHOIS = False
ENABLE_CONTENT = False

SUSPICIOUS_KEYWORDS = [
    "login", "signin", "sign-in", "secure", "account", "update", "verify",
    "confirm", "password", "bank", "paypal", "icloud", "microsoft", "office",
    "auth", "support", "billing", "invoice", "urgent"
]

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CI601-Feature-Extractor/1.0)"}
REQUEST_TIMEOUT = 8 
socket.setdefaulttimeout(8)
EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=None)

if ENABLE_CONTENT and InsecureRequestWarning is not None:
    warnings.simplefilter("ignore", InsecureRequestWarning)

# -------------------------
# Helper functions
# -------------------------
def shannon_entropy(s: str) -> float:
    if not s: return 0.0
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    ent = 0.0
    length = len(s)
    for c in counts.values():
        p = c / length
        ent -= p * math.log2(p)
    return ent

def count_keywords(text: str, keywords) -> int:
    if not text: return 0
    t = text.lower()
    return sum(1 for k in keywords if k in t)

def clean_url(url: str) -> str:
    if not isinstance(url, str): return ""
    u = url.strip()
    if not u: return ""
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", u):
        u = "http://" + u
    return u

# -------------------------
# URL lexical + structural features
# -------------------------
def extract_url_features(url: str) -> dict:
    """
    Core extraction logic used for both training and real-time inference.
    """
    default = {
        "url_length": 0, "host_length": 0, "path_length": 0, "query_length": 0,
        "num_digits": 0, "num_letters": 0, "num_special_chars": 0, "num_dots": 0,
        "num_hyphens": 0, "has_ip_in_host": 0, "subdomain_depth": 0,
        "suspicious_keyword_count": 0, "url_entropy": 0.0, "is_https": 0, "url_parse_error": 1,
    }

    try:
        parsed = urlparse(url)
    except Exception:
        return default

    host = parsed.netloc
    full = url or ""

    try:
        ext = EXTRACTOR(url)
        subdomain = ext.subdomain or ""
        subdomain_depth = 0 if not subdomain else len([p for p in subdomain.split(".") if p])
    except Exception:
        subdomain_depth = 0

    return {
        "url_length": len(full),
        "host_length": len(host),
        "path_length": len(parsed.path or ""),
        "query_length": len(parsed.query or ""),
        "num_digits": sum(ch.isdigit() for ch in full),
        "num_letters": sum(ch.isalpha() for ch in full),
        "num_special_chars": sum((not ch.isalnum()) for ch in full),
        "num_dots": full.count("."),
        "num_hyphens": full.count("-"),
        "has_ip_in_host": 1 if re.search(r"^\d{1,3}(\.\d{1,3}){3}$", host) else 0,
        "subdomain_depth": subdomain_depth,
        "suspicious_keyword_count": count_keywords(full, SUSPICIOUS_KEYWORDS),
        "url_entropy": shannon_entropy(full),
        "is_https": 1 if parsed.scheme.lower() == "https" else 0,
        "url_parse_error": 0,
    }

# -------------------------
# Normalization Logic
# -------------------------
def normalise_numeric(df: pd.DataFrame, method: str = "zscore"):
    """
    Normalizes numeric features and returns the fitted scaler for persistence.
    """
    df_out = df.copy()
    numeric_cols = df_out.select_dtypes(include=[np.number]).columns.tolist()

    for c in numeric_cols:
        med = df_out[c].median()
        df_out[c] = df_out[c].fillna(med)

    scaler = StandardScaler() if method == "zscore" else MinMaxScaler()
    df_out[numeric_cols] = scaler.fit_transform(df_out[numeric_cols])
    
    return df_out, scaler

# -------------------------
# Main Execution Block
# -------------------------
if __name__ == "__main__":
    # DEFENSIVE PROGRAMMING: Ensure the 'models' directory exists before saving
    os.makedirs("models", exist_ok=True) #

    print(f"Step 1: Loading dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df["url"] = df["url"].astype(str).apply(clean_url)

    print("Step 2: Extracting raw lexical features...")
    rows = [extract_url_features(u) for u in df["url"]]
    features_raw = pd.DataFrame(rows)

    print("Step 3: Normalizing features and fitting scaler...")
    features_norm, fitted_scaler = normalise_numeric(features_raw, method="zscore") #
    
    # Attach labels for the CSV export
    features_raw["label"] = df["label"].values
    features_norm["label"] = df["label"].values

    print("Step 4: Persisting artifacts to disk...")
    features_raw.to_csv("features_raw.csv", index=False)
    features_norm.to_csv("features_normalised.csv", index=False)
    
    # SAVE THE SCALER: This is required for the Web API to function correctly
    joblib.dump(fitted_scaler, "models/scaler.joblib") #
    
    print("Process Complete. Files generated:")
    print(" - features_raw.csv")
    print(" - features_normalised.csv")
    print(" - models/scaler.joblib")
