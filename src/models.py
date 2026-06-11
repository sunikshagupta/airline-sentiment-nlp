"""
models.py
=========
Step 3 of the NLP pipeline — Layer 1: Classical ML.

Models trained and evaluated:
  1. TF-IDF + Logistic Regression  (baseline + SHAP explainability)
  2. TF-IDF + LinearSVC            (usually slightly stronger)

Outputs saved to outputs/:
  - classification reports (printed)
  - confusion matrices
  - SHAP word importance charts
  - model comparison summary

Run:
    python src/models.py
"""

import logging
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import shap

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
)
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s")

LABEL_NAMES = ["negative", "neutral", "positive"]
PALETTE     = {"negative": "#E24B4A", "neutral": "#888780", "positive": "#1D9E75"}

# ── Load data ─────────────────────────────────────────────────────────────────

def load_splits():
    train = pd.read_csv("data/train.csv")
    val   = pd.read_csv("data/val.csv")
    test  = pd.read_csv("data/test.csv")
    logging.info(f"Loaded splits → train={len(train):,} | val={len(val):,} | test={len(test):,}")
    return train, val, test

# ── TF-IDF vectorizer ─────────────────────────────────────────────────────────

def build_tfidf():
    """
    TF-IDF with bigrams:
      - max_features=15,000  : keep the 15k most informative word/bigram features
      - ngram_range=(1,2)    : include single words AND two-word phrases
      - min_df=2             : ignore terms appearing in fewer than 2 tweets (noise)
      - sublinear_tf=True    : apply log(1+tf) to dampen high-frequency terms
    """
    return TfidfVectorizer(
        max_features=15_000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
    )

# ── Save helper ───────────────────────────────────────────────────────────────

def save_fig(fig, name: str):
    path = Path("outputs") / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved → {path}")

# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, model_name: str, filename: str):
    cm  = confusion_matrix(y_true, y_pred)
    pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", ax=axes[0],
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
                cmap="Blues", linewidths=0.5, linecolor="white")
    axes[0].set_title(f"{model_name}\nConfusion Matrix (counts)", fontweight="bold")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")

    # Row-normalised %
    sns.heatmap(pct, annot=True, fmt=".1f", ax=axes[1],
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
                cmap="Blues", linewidths=0.5, linecolor="white",
                vmin=0, vmax=100)
    axes[1].set_title(f"{model_name}\nConfusion Matrix (% of actual class)", fontweight="bold")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")

    fig.tight_layout()
    save_fig(fig, filename)

# ── Classification report printer ────────────────────────────────────────────

def print_report(y_true, y_pred, model_name: str):
    print(f"\n{'='*60}")
    print(f"CLASSIFICATION REPORT — {model_name}")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES))
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="weighted")
    print(f"  Accuracy (overall) : {acc*100:.2f}%")
    print(f"  F1 weighted        : {f1*100:.2f}%")
    return acc, f1

# ── Model 1: Logistic Regression ──────────────────────────────────────────────

def train_logistic_regression(train, val, test):
    logging.info("Training Model 1: TF-IDF + Logistic Regression ...")

    tfidf = build_tfidf()
    X_train = tfidf.fit_transform(train["clean_text"])
    X_val   = tfidf.transform(val["clean_text"])
    X_test  = tfidf.transform(test["clean_text"])

    # class_weight='balanced' compensates for the 63/21/16 imbalance
    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        random_state=42,
    )
    lr.fit(X_train, train["label"])

    # Validate
    val_preds  = lr.predict(X_val)
    test_preds = lr.predict(X_test)

    logging.info("Validation results:")
    val_acc, val_f1 = print_report(val["label"], val_preds, "LR — Validation")

    logging.info("Test results:")
    test_acc, test_f1 = print_report(test["label"], test_preds, "LR — Test")

    # Confusion matrix
    plot_confusion_matrix(test["label"], test_preds,
                          "Logistic Regression", "07_cm_logistic_regression.png")

    # Save model + vectorizer
    with open("outputs/tfidf_lr_model.pkl", "wb") as f:
        pickle.dump({"tfidf": tfidf, "model": lr, "label_names": LABEL_NAMES}, f)
    logging.info("Model saved → outputs/tfidf_lr_model.pkl")

    return lr, tfidf, test_preds, test_acc, test_f1

# ── SHAP explainability ───────────────────────────────────────────────────────

def plot_shap(lr, tfidf, test, n_samples: int = 300):
    """
    SHAP LinearExplainer shows which words push predictions
    toward each sentiment class.
    Answers: "WHY did the model call this tweet negative?"
    """
    logging.info("Running SHAP analysis (this takes ~30 seconds) ...")

    X_test_sparse = tfidf.transform(test["clean_text"])
    feature_names = tfidf.get_feature_names_out()

    # Use a sample for speed
    idx = np.random.RandomState(42).choice(X_test_sparse.shape[0],
                                           size=min(n_samples, X_test_sparse.shape[0]),
                                           replace=False)
    X_sample = X_test_sparse[idx]

    explainer   = shap.LinearExplainer(lr, X_train_bg := X_test_sparse,
                                       feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_sample)

    # One chart per class
    class_labels = ["Negative", "Neutral", "Positive"]
    filenames    = ["08a_shap_negative.png", "08b_shap_neutral.png", "08c_shap_positive.png"]

    for i, (label, fname) in enumerate(zip(class_labels, filenames)):
        fig, ax = plt.subplots(figsize=(10, 6))
        sv = shap_values[i] if isinstance(shap_values, list) else shap_values[:, :, i]

        mean_abs = np.abs(sv).mean(axis=0)
        top_idx  = np.argsort(mean_abs)[-20:][::-1]
        top_feats = feature_names[top_idx]
        top_vals  = mean_abs[top_idx]

        colors = ["#E24B4A" if label=="Negative" else
                  "#888780" if label=="Neutral"  else "#1D9E75"] * 20

        ax.barh(range(len(top_feats)), top_vals[::-1], color=colors[::-1])
        ax.set_yticks(range(len(top_feats)))
        ax.set_yticklabels(top_feats[::-1], fontsize=9)
        ax.set_title(f"Top 20 Words Driving '{label}' Predictions\n(mean |SHAP value|)",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Mean |SHAP value| — higher = more influential")
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        save_fig(fig, fname)

    logging.info("SHAP charts saved → outputs/08a/b/c_shap_*.png")

# ── Model 2: LinearSVC ────────────────────────────────────────────────────────

def train_linear_svc(train, val, test):
    logging.info("Training Model 2: TF-IDF + LinearSVC ...")

    tfidf = build_tfidf()
    X_train = tfidf.fit_transform(train["clean_text"])
    X_val   = tfidf.transform(val["clean_text"])
    X_test  = tfidf.transform(test["clean_text"])

    # CalibratedClassifierCV wraps LinearSVC to give probability estimates
    svc = CalibratedClassifierCV(
        LinearSVC(class_weight="balanced", max_iter=2000, C=1.0, random_state=42)
    )
    svc.fit(X_train, train["label"])

    val_preds  = svc.predict(X_val)
    test_preds = svc.predict(X_test)

    logging.info("Validation results:")
    val_acc, val_f1 = print_report(val["label"], val_preds, "LinearSVC — Validation")

    logging.info("Test results:")
    test_acc, test_f1 = print_report(test["label"], test_preds, "LinearSVC — Test")

    plot_confusion_matrix(test["label"], test_preds,
                          "LinearSVC", "09_cm_linearsvc.png")

    with open("outputs/tfidf_svc_model.pkl", "wb") as f:
        pickle.dump({"tfidf": tfidf, "model": svc, "label_names": LABEL_NAMES}, f)
    logging.info("Model saved → outputs/tfidf_svc_model.pkl")

    return svc, tfidf, test_preds, test_acc, test_f1

# ── Model comparison chart ────────────────────────────────────────────────────

def plot_model_comparison(results: dict):
    """
    Bar chart comparing accuracy and weighted F1 across all models.
    This goes straight into the README.
    """
    models   = list(results.keys())
    accs     = [results[m]["accuracy"] * 100 for m in models]
    f1s      = [results[m]["f1"]       * 100 for m in models]

    x   = np.arange(len(models))
    w   = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - w/2, accs, w, label="Accuracy",    color="#534AB7", alpha=0.85)
    bars2 = ax.bar(x + w/2, f1s,  w, label="Weighted F1", color="#1D9E75", alpha=0.85)

    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(60, 100)
    ax.set_title("Layer 1 Model Comparison — Test Set Performance",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Score (%)")
    ax.legend()
    ax.spines[["top","right"]].set_visible(False)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    fig.tight_layout()
    save_fig(fig, "10_model_comparison.png")

    # Print summary table
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"  {'Model':<30} {'Accuracy':>10} {'F1 (weighted)':>15}")
    print(f"  {'-'*55}")
    for model in models:
        acc = results[model]["accuracy"] * 100
        f1  = results[model]["f1"]       * 100
        print(f"  {model:<30} {acc:>9.2f}%  {f1:>14.2f}%")

# ── Main ─────────────────────────────────────────────────────────────────────

def run_models():
    Path("outputs").mkdir(exist_ok=True)

    train, val, test = load_splits()

    # Model 1 — Logistic Regression
    lr, tfidf_lr, lr_preds, lr_acc, lr_f1 = train_logistic_regression(train, val, test)

    # SHAP explainability on LR
    plot_shap(lr, tfidf_lr, test)

    # Model 2 — LinearSVC
    svc, tfidf_svc, svc_preds, svc_acc, svc_f1 = train_linear_svc(train, val, test)

    # Comparison
    results = {
        "Logistic Regression": {"accuracy": lr_acc,  "f1": lr_f1},
        "LinearSVC":           {"accuracy": svc_acc, "f1": svc_f1},
    }
    plot_model_comparison(results)

    logging.info("✅ Layer 1 modelling complete.")
    logging.info("   Next step: Layer 2 — run src/models_transformer.py")

if __name__ == "__main__":
    run_models()
