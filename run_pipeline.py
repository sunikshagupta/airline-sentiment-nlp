"""
run_pipeline.py
===============
Master script — runs the full Layer 1 pipeline in order:
  1. Preprocessing  (data/tweets_clean.csv + splits)
  2. EDA            (outputs/01–06 charts)
  3. Modelling      (outputs/07–10 charts + saved models)

Run from your project root:
    python run_pipeline.py

Or run individual steps:
    python src/preprocessing.py
    python src/eda.py
    python src/models.py
"""

import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s")


def main():
    start = time.time()
    Path("outputs").mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("  AIRLINE SENTIMENT NLP — LAYER 1 PIPELINE")
    print("=" * 60)

    # ── Step 1: Preprocessing ────────────────────────────────────────────
    print("\n📦  STEP 1/3 — Preprocessing")
    print("-" * 40)
    from src.preprocessing import run_preprocessing
    df, train, val, test = run_preprocessing()

    # ── Step 2: EDA ──────────────────────────────────────────────────────
    print("\n📊  STEP 2/3 — Exploratory Data Analysis")
    print("-" * 40)
    from src.eda import run_eda
    run_eda()

    # ── Step 3: Modelling ────────────────────────────────────────────────
    print("\n🤖  STEP 3/3 — Classical ML Models")
    print("-" * 40)
    from src.models import run_models
    run_models()

    # ── Done ─────────────────────────────────────────────────────────────
    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"  ✅  PIPELINE COMPLETE in {elapsed:.0f} seconds")
    print("=" * 60)
    print("\n  Files saved:")
    print("    data/tweets_clean.csv   ← cleaned dataset")
    print("    data/train.csv          ← training split")
    print("    data/val.csv            ← validation split")
    print("    data/test.csv           ← test split")
    print("    outputs/01–06_*.png     ← EDA charts")
    print("    outputs/07–10_*.png     ← model evaluation charts")
    print("    outputs/tfidf_lr_model.pkl   ← saved LR model")
    print("    outputs/tfidf_svc_model.pkl  ← saved SVC model")
    print("\n  Next steps:")
    print("    → Add Layer 2: python src/models_transformer.py")
    print("    → Build the app: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
