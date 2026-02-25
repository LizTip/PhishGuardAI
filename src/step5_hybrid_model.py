import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from scipy.sparse import hstack, csr_matrix

# -----------------------------------
# Step 5.1: Load original dataset
# -----------------------------------
DATA_PATH = r"balanced_urls_dataset.csv"

print(f"Loading raw dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

if "url" not in df.columns or "label" not in df.columns:
    raise ValueError("Dataset must contain 'url' and 'label' columns.")

# -----------------------------------
# Step 5.2: Train/Test split
# -----------------------------------
X_url = df["url"].astype(str)
y = df["label"]

X_train_url, X_test_url, y_train, y_test = train_test_split(
    X_url,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Train/Test split complete.")

# -----------------------------------
# Step 5.3: TF-IDF Character n-grams
# -----------------------------------
tfidf = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=5000
)

X_train_tfidf = tfidf.fit_transform(X_train_url)
X_test_tfidf = tfidf.transform(X_test_url)

print("TF-IDF features created.")

# -----------------------------------
# Step 5.4: Load numeric features
# -----------------------------------
numeric_df = pd.read_csv("features_normalised.csv")
X_numeric = numeric_df.drop(columns=["label"])

X_train_num, X_test_num, _, _ = train_test_split(
    X_numeric,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train_num_sparse = csr_matrix(X_train_num.values)
X_test_num_sparse = csr_matrix(X_test_num.values)

# -----------------------------------
# Step 5.5: Combine numeric + TF-IDF features
# -----------------------------------
X_train_hybrid = hstack([X_train_num_sparse, X_train_tfidf])
X_test_hybrid = hstack([X_test_num_sparse, X_test_tfidf])

print("Hybrid feature matrix created.")

# -----------------------------------
# Step 5.6: Train Hybrid Logistic Regression
# -----------------------------------
hybrid_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

hybrid_model.fit(X_train_hybrid, y_train)
print("Hybrid model trained.")

# -----------------------------------
# NEW: Persistence Logic for Web Deployment
# -----------------------------------
# Create a models directory to stay organised
if not os.path.exists("models"):
    os.makedirs("models")

print("Saving model artifacts for deployment...")

# Save the trained Hybrid Model
joblib.dump(hybrid_model, "models/hybrid_model.joblib")

# Save the TF-IDF Vectorizer
joblib.dump(tfidf, "models/tfidf_vectorizer.joblib")

# Save the column names to ensure consistent feature order in the API
joblib.dump(X_numeric.columns.tolist(), "models/feature_names.joblib")

# NOTE: Ensure to save 'scaler' object here if you re-run scaling script!
# joblib.dump(scaler, "models/scaler.joblib")

print("Artifacts saved in /models/ folder.")

# -----------------------------------
# Step 5.7: Evaluate Hybrid Model
# -----------------------------------
y_pred = hybrid_model.predict(X_test_hybrid)
y_prob = hybrid_model.predict_proba(X_test_hybrid)[:, 1]

print("\nHybrid Logistic Regression Results:")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC–AUC  : {roc_auc_score(y_test, y_prob):.4f}")

# -----------------------------------
# Step 5.8: Statistical Strengthening (k-Fold CV)
# -----------------------------------
from sklearn.model_selection import StratifiedKFold, cross_val_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Hybrid Cross-Validation
tfidf_full = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=5000)
X_tfidf_full = tfidf_full.fit_transform(df["url"].astype(str))
X_numeric_sparse = csr_matrix(X_numeric.values)
X_hybrid_full = hstack([X_numeric_sparse, X_tfidf_full])

hybrid_cv_scores = cross_val_score(
    hybrid_model, X_hybrid_full, y, cv=skf, scoring="roc_auc", n_jobs=-1
)

print(f"\nHybrid Mean ROC–AUC: {hybrid_cv_scores.mean():.4f}")


# -----------------------------------
# Step 5.9: Bootstrapped 95% Confidence Intervals
# -----------------------------------
def bootstrap_ci(y_true, y_pred_labels, y_pred_probs, n_boot=500, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    auc_list = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        auc_list.append(roc_auc_score(y_true.iloc[idx], y_pred_probs[idx]))

    mean_val = np.mean(auc_list)
    return mean_val, np.percentile(auc_list, 2.5), np.percentile(auc_list, 97.5)


mean, lo, hi = bootstrap_ci(y_test.reset_index(drop=True), y_pred, y_prob)
print(f"ROC–AUC 95% CI: {mean:.4f} [{lo:.4f}, {hi:.4f}]")