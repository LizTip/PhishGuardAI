import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# -----------------------------------
# Step 4.1: Load engineered features and create X/y
# -----------------------------------

# This CSV is produced by my feature engineering script
# It contains 15 numeric URL features + 1 label column
FEATURES_PATH = r"../data/features_normalised.csv"

# Load the engineered dataset
print(f"Loading features from: {FEATURES_PATH}")
df = pd.read_csv(FEATURES_PATH)

# Validate that the label column exists (required for supervised learning)
if "label" not in df.columns:
    raise ValueError(f"'label' column not found. Columns found: {list(df.columns)}")

# Split into:
# X = feature matrix (all columns except label)
# y = target vector (label column: 0=benign, 1=phishing)
X = df.drop(columns=["label"])
y = df["label"]

# Print dataset summaries (for logging and report evidence)
print("Dataset shape:", df.shape)
print("Feature matrix (X) shape:", X.shape)
print("Labels (y) distribution:\n", y.value_counts(dropna=False))

# Safety check: baseline Logistic Regression requires numeric features only
non_numeric = X.select_dtypes(exclude=["number"]).columns.tolist()
if non_numeric:
    raise ValueError(f"Non-numeric feature columns found in X: {non_numeric}")

print("Step 4.1 complete: X and y ready.\n")

# -----------------------------------
# Step 4.2: Train/Test split (baseline)
# -----------------------------------

# Hold out 20% of the data for final testing
TEST_SIZE = 0.20

# Fixed seed ensures reproducible results across runs
RANDOM_STATE = 42

# Split the dataset into training and test sets
# stratify=y ensures class balance is preserved in both splits
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# Print split summaries (for verification + report table)
print("Step 4.2 complete: Train/Test split created.")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train distribution:\n", y_train.value_counts())
print("y_test distribution:\n", y_test.value_counts())

# -----------------------------------
# Step 4.3: Train baseline Logistic Regression
# -----------------------------------

# Logistic Regression is a strong, interpretable baseline model
# class_weight="balanced" reduces bias towards the majority class (for security datasets)
print("\nTraining baseline Logistic Regression model...")

baseline_model = LogisticRegression(
    max_iter=1000,           # Increase iterations to ensure convergence
    class_weight="balanced", # Adjust weights inversely proportional to class frequency
    random_state=42          # Reproducibility
)

# Train the model on the training split only
baseline_model.fit(X_train, y_train)

print("Step 4.3 complete: Baseline Logistic Regression trained.")

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# -----------------------------------
# Step 4.4: Baseline model evaluation (test set)
# -----------------------------------

print("\nEvaluating baseline Logistic Regression model on the test set...")

# Predict class labels for the test set (0 = benign, 1 = phishing)
y_pred = baseline_model.predict(X_test)

# Predict phishing probabilities for the test set (needed for ROC–AUC)
# [:, 1] selects the probability of class 1 (phishing)
y_prob = baseline_model.predict_proba(X_test)[:, 1]

# --- Core metrics (baseline requirements) ---
# Accuracy: overall correctness
accuracy = accuracy_score(y_test, y_pred)

# Precision: of everything predicted as phishing, how many were truly phishing?
precision = precision_score(y_test, y_pred, pos_label=1)

# Recall: of all true phishing URLs, how many were detected?
recall = recall_score(y_test, y_pred, pos_label=1)

# ROC–AUC: discrimination ability across all decision thresholds
roc_auc = roc_auc_score(y_test, y_prob)

# --- Confusion matrix ---
# Format:
# [[TN, FP],
#  [FN, TP]]
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# --- Print headline results ---
print("\n[M01] Baseline Logistic Regression: Performance Metrics")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"ROC–AUC  : {roc_auc:.4f}")

# --- Print confusion matrix + error counts ---
print("\nConfusion Matrix [[TN, FP], [FN, TP]]:")
print(cm)

print("\nError breakdown:")
print(f"True Negatives (TN): {tn}")  # benign correctly classified
print(f"False Positives (FP): {fp}") # benign incorrectly flagged as phishing
print(f"False Negatives (FN): {fn}") # phishing missed (most costly in security)
print(f"True Positives (TP): {tp}")  # phishing correctly detected

# --- Optional detailed report (add to appendix) ---
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))


