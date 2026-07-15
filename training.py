
# 1) IMPORT LIBRARIES
# ---------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

# ---------------------------------------------------------------
# 2) LOAD DATA
# ---------------------------------------------------------------
df = pd.read_csv("data/telco_churn.csv")
print("Shape:", df.shape)
print(df.info())
print(df.describe(include="all").T)

# ---------------------------------------------------------------
# 3) EXPLORE DATA (EDA) -> save plots as PNG screenshots
# ---------------------------------------------------------------
# 3.1 Class balance
plt.figure(figsize=(5, 4))
df["Churn"].value_counts().plot(kind="bar", color=["#2E86AB", "#E63946"])
plt.title("Churn Class Distribution")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plots/01_class_balance.png")
plt.close()

# 3.2 Tenure distribution by churn
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack", bins=20)
plt.title("Tenure Distribution by Churn")
plt.tight_layout()
plt.savefig("plots/02_tenure_distribution.png")
plt.close()

# 3.3 Monthly charges by churn
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges by Churn")
plt.tight_layout()
plt.savefig("plots/03_monthly_charges_boxplot.png")
plt.close()

# 3.4 Churn rate by contract type
plt.figure(figsize=(6, 4))
churn_by_contract = df.groupby("Contract")["Churn"].apply(lambda x: (x == "Yes").mean())
churn_by_contract.plot(kind="bar", color="#F4A261")
plt.title("Churn Rate by Contract Type")
plt.ylabel("Churn Rate")
plt.tight_layout()
plt.savefig("plots/04_churn_by_contract.png")
plt.close()

# 3.5 Correlation heatmap (numeric features)
plt.figure(figsize=(5, 4))
numeric_df = df[["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]]
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/05_correlation_heatmap.png")
plt.close()

print("EDA plots saved in /plots")

# ---------------------------------------------------------------
# 4) PREPROCESSING
# ---------------------------------------------------------------
data = df.copy()

# Encode target
data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})

# Encode categorical features
categorical_cols = data.select_dtypes(include="object").columns.tolist()
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

X = data.drop("Churn", axis=1)
y = data["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# 5) MODEL TRAINING
# ---------------------------------------------------------------
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)

rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
rf.fit(X_train, y_train)  # tree model doesn't need scaling

models = {"Logistic Regression": (log_reg, X_test_scaled), "Random Forest": (rf, X_test)}

# ---------------------------------------------------------------
# 6) EVALUATION
# ---------------------------------------------------------------
results = []
for name, (model, X_te) in models.items():
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results.append([name, acc, prec, rec, f1, auc])
    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    fname = name.lower().replace(" ", "_")
    plt.savefig(f"plots/06_confusion_matrix_{fname}.png")
    plt.close()

results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
print("\n=== Model Comparison ===")
print(results_df.to_string(index=False))
results_df.to_csv("model/model_comparison.csv", index=False)

# ROC curve comparison
plt.figure(figsize=(6, 5))
for name, (model, X_te) in models.items():
    y_proba = model.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("plots/07_roc_curve.png")
plt.close()

# ---------------------------------------------------------------
# 7) SAVE BEST MODEL (Random Forest, no scaling needed -> simpler deployment)
# ---------------------------------------------------------------
joblib.dump(rf, "model/churn_model.pkl")
joblib.dump(encoders, "model/encoders.pkl")
joblib.dump(list(X.columns), "model/feature_columns.pkl")

print("\nModel + encoders + feature columns saved in /model")
print("Pipeline complete.")
