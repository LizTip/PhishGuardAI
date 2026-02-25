import re
import math
import numpy as np
import pandas as pd
import tldextract
import whois
import requests

from bs4 import BeautifulSoup
from urllib.parse import urlparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# -------------------------
# Config
# -------------------------
DATA_PATH = r"C:\Users\lizti\OneDrive\Documents\BrightonUni\TheComputingProject\Step 2 Data Sets\balanced_urls_dataset.csv"

SUSPICIOUS_KEYWORDS = [
    "login", "signin", "sign-in", "secure", "account", "update", "verify",
    "confirm", "password", "bank", "paypal", "icloud", "microsoft", "office",
    "auth", "support", "billing", "invoice", "urgent"
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CI601-Feature-Extractor/1.0)"
}

REQUEST_TIMEOUT = 8  # seconds


# -------------------------
# Helper functions
# -------------------------
def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
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
    if not text:
        return 0
    t = text.lower()
    return sum(1 for k in keywords if k in t)

def clean_url(url: str) -> str:
    """
    Ensures the URL has a scheme so urlparse() + requests work reliably.
    As dataset includes values like 'www.site.com/page' which need 'http://'.
    """
    if not isinstance(url, str):
        return ""
    u = url.strip()
    if not u:
        return ""
    # If it starts with www. or looks like a domain but has no scheme, add http://
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", u):
        u = "http://" + u
    return u


# -------------------------
# URL lexical + structural features
# -------------------------
def extract_url_features(url: str) -> dict:
    parsed = urlparse(url)
    host = parsed.netloc
    path = parsed.path or ""
    query = parsed.query or ""
    full = url or ""

    num_digits = sum(ch.isdigit() for ch in full)
    num_letters = sum(ch.isalpha() for ch in full)
    num_special = sum((not ch.isalnum()) for ch in full)

    num_dots = full.count(".")
    num_hyphens = full.count("-")

    has_ip = 1 if re.search(r"^\d{1,3}(\.\d{1,3}){3}$", host) else 0

    ext = tldextract.extract(url)
    subdomain = ext.subdomain or ""
    subdomain_depth = 0 if not subdomain else len([p for p in subdomain.split(".") if p])

    suspicious_kw = count_keywords(full, SUSPICIOUS_KEYWORDS)
    ent = shannon_entropy(full)

    is_https = 1 if parsed.scheme.lower() == "https" else 0

    return {
        "url_length": len(full),
        "host_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "num_digits": num_digits,
        "num_letters": num_letters,
        "num_special_chars": num_special,
        "num_dots": num_dots,
        "num_hyphens": num_hyphens,
        "has_ip_in_host": has_ip,
        "subdomain_depth": subdomain_depth,
        "suspicious_keyword_count": suspicious_kw,
        "url_entropy": ent,
        "is_https": is_https,
    }


# -------------------------
# WHOIS features (domain trust signals)
# -------------------------
def extract_whois_features(url: str) -> dict:
    ext = tldextract.extract(url)
    domain = ".".join([p for p in [ext.domain, ext.suffix] if p])

    out = {
        "whois_available": 0,
        "domain_age_days": np.nan,
        "days_to_expiry": np.nan,
        "registrar_present": 0,
    }

    if not domain:
        return out

    try:
        w = whois.whois(domain)
        out["whois_available"] = 1

        creation = w.creation_date
        expiry = w.expiration_date

        if isinstance(creation, list):
            creation = creation[0]
        if isinstance(expiry, list):
            expiry = expiry[0]

        now = pd.Timestamp.utcnow()

        if creation:
            out["domain_age_days"] = (now - pd.Timestamp(creation)).days
        if expiry:
            out["days_to_expiry"] = (pd.Timestamp(expiry) - now).days

        registrar = w.registrar
        out["registrar_present"] = 1 if registrar else 0

    except Exception:
        # WHOIS can fail often; that’s expected
        pass

    return out


# -------------------------
# Content-based features (HTML)
# -------------------------
def fetch_html(url: str) -> str | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        ct = (r.headers.get("Content-Type") or "").lower()
        if "text/html" not in ct:
            return None
        return r.text
    except Exception:
        return None

def extract_content_features(url: str) -> dict:
    out = {
        "content_available": 0,
        "title_length": np.nan,
        "title_suspicious_keyword_count": np.nan,
        "visible_text_suspicious_keyword_count": np.nan,
        "num_forms": np.nan,
        "has_password_input": np.nan,
        "num_external_scripts": np.nan,
        "num_external_links": np.nan,
    }

    html = fetch_html(url)
    if not html:
        return out

    out["content_available"] = 1
    soup = BeautifulSoup(html, "html.parser")

    title = (soup.title.string.strip() if soup.title and soup.title.string else "")
    out["title_length"] = len(title)
    out["title_suspicious_keyword_count"] = count_keywords(title, SUSPICIOUS_KEYWORDS)

    text = soup.get_text(" ", strip=True)
    out["visible_text_suspicious_keyword_count"] = count_keywords(text, SUSPICIOUS_KEYWORDS)

    forms = soup.find_all("form")
    out["num_forms"] = len(forms)

    out["has_password_input"] = 1 if soup.find("input", {"type": "password"}) else 0

    parsed = urlparse(url)
    base_host = parsed.netloc

    scripts = soup.find_all("script", src=True)
    ext_scripts = 0
    for s in scripts:
        src = s.get("src", "")
        h = urlparse(src).netloc
        if h and h != base_host:
            ext_scripts += 1
    out["num_external_scripts"] = ext_scripts

    links = soup.find_all("a", href=True)
    ext_links = 0
    for a in links:
        href = a.get("href", "")
        h = urlparse(href).netloc
        if h and h != base_host:
            ext_links += 1
    out["num_external_links"] = ext_links

    return out


# -------------------------
# Pipeline: build DataFrame + normalise
# -------------------------
def build_feature_dataframe(urls: pd.Series, do_whois=True, do_content=True) -> pd.DataFrame:
    rows = []
    for raw_url in urls.fillna(""):
        url = clean_url(raw_url)  # IMPORTANT: fixes missing http://
        feat = {}
        feat.update(extract_url_features(url))

        if do_whois:
            feat.update(extract_whois_features(url))

        if do_content:
            feat.update(extract_content_features(url))

        rows.append(feat)

    return pd.DataFrame(rows)

def normalise_numeric(df: pd.DataFrame, method: str = "zscore") -> pd.DataFrame:
    df_out = df.copy()
    numeric_cols = df_out.select_dtypes(include=[np.number]).columns.tolist()

    for c in numeric_cols:
        med = df_out[c].median()
        df_out[c] = df_out[c].fillna(med)

    if method == "zscore":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("method must be 'zscore' or 'minmax'")

    df_out[numeric_cols] = scaler.fit_transform(df_out[numeric_cols])
    return df_out


# -------------------------
# Main: load dataset -> extract -> normalise -> save
# -------------------------
if __name__ == "__main__":
    # 1) Load dataset
    df = pd.read_csv(DATA_PATH)

    # 2) Basic validation
    required_cols = {"url", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {required_cols}. Found: {list(df.columns)}")

    # 3) Clean URL column (ensures scheme exists)
    df["url"] = df["url"].astype(str).apply(clean_url)

    # 4) Feature extraction (WHOIS + Content ON as you requested)
    features = build_feature_dataframe(df["url"], do_whois=True, do_content=True)

    # 5) Add labels back in
    features["label"] = df["label"].values

    # 6) Normalise (z-score is the best default for LR/SVM)
    features_norm = normalise_numeric(features.drop(columns=["label"]), method="zscore")
    features_norm["label"] = features["label"]

    # 7) Save outputs
    features.to_csv("features_raw.csv", index=False)
    features_norm.to_csv("features_normalised.csv", index=False)

    print("Saved: features_raw.csv")
    print("Saved: features_normalised.csv")
    print("Done.")
