"""
preprocessing.py
================
Step 1 of the NLP pipeline.
Loads raw Tweets.csv, cleans text, encodes labels,
and saves stratified train/val/test splits to disk.

Run:
    python src/preprocessing.py
"""

import re
import logging
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s")

# ── Constants ────────────────────────────────────────────────────────────────

RAW_PATH   = Path("data/Tweets.csv")
CLEAN_PATH = Path("data/tweets_clean.csv")

LABEL_MAP  = {"negative": 0, "neutral": 1, "positive": 2}

KEEP_COLS  = [
    "tweet_id", "text", "airline_sentiment",
    "airline_sentiment_confidence", "airline",
    "negativereason", "negativereason_confidence",
    "tweet_created", "retweet_count",
]

# ── 1. Load ──────────────────────────────────────────────────────────────────

def load_data(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    logging.info(f"Loaded {len(df):,} rows | {len(df.columns)} columns")
    return df

# ── 2. Select columns ────────────────────────────────────────────────────────

def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df[KEEP_COLS].copy()
    logging.info(f"Kept {len(df.columns)} columns: {df.columns.tolist()}")
    return df

# ── 3. Clean text ────────────────────────────────────────────────────────────

def clean_tweet(text: str) -> str:
    """
    Light cleaning tailored to Twitter text:
      - Remove @mentions, URLs, RT marker, # symbol
      - Keep exclamation marks (strong sentiment signal)
      - Remove everything else that isn't a letter or space
      - Lowercase and strip extra whitespace
    """
    text = re.sub(r"@\w+", "", text)            # @mentions
    text = re.sub(r"http\S+|www\S+", "", text)  # URLs
    text = re.sub(r"\bRT\b", "", text)           # retweet marker
    text = re.sub(r"#", "", text)               # hashtag symbol
    text = re.sub(r"[^a-zA-Z!\s]", "", text)    # keep letters + !
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

def clean_text_column(df: pd.DataFrame) -> pd.DataFrame:
    df["clean_text"] = df["text"].apply(clean_tweet)
    before = len(df)
    df = df[df["clean_text"].str.len() > 5].copy()
    logging.info(f"Text cleaned | Dropped {before - len(df)} near-empty tweets | {len(df):,} remaining")
    return df

# ── 4. Encode labels ─────────────────────────────────────────────────────────

def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["airline_sentiment"].isin(LABEL_MAP)].copy()
    df["label"] = df["airline_sentiment"].map(LABEL_MAP)
    logging.info(f"Labels encoded | negative=0 | neutral=1 | positive=2")

    counts = df["airline_sentiment"].value_counts()
    total  = len(df)
    logging.info("Class distribution:")
    for sentiment, count in counts.items():
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        logging.info(f"  {sentiment:<12} {count:>6,}  ({pct:.1f}%)  {bar}")
    return df

# ── 5. Stratified split ──────────────────────────────────────────────────────

def split_data(df: pd.DataFrame, random_state: int = 42):
    """80% train / 10% val / 10% test — all stratified on label."""
    train_val, test = train_test_split(
        df, test_size=0.10, stratify=df["label"], random_state=random_state
    )
    train, val = train_test_split(
        train_val, test_size=0.111, stratify=train_val["label"], random_state=random_state
    )
    logging.info(f"Split → train={len(train):,} | val={len(val):,} | test={len(test):,}")
    return train, val, test

# ── 6. Save ──────────────────────────────────────────────────────────────────

def save_data(df, train, val, test):
    df.to_csv("data/tweets_clean.csv", index=False)
    train.to_csv("data/train.csv",     index=False)
    val.to_csv("data/val.csv",         index=False)
    test.to_csv("data/test.csv",       index=False)
    logging.info("Saved: data/tweets_clean.csv | train.csv | val.csv | test.csv")

# ── Main ─────────────────────────────────────────────────────────────────────

def run_preprocessing():
    df    = load_data()
    df    = select_columns(df)
    df    = clean_text_column(df)
    df    = encode_labels(df)
    train, val, test = split_data(df)
    save_data(df, train, val, test)
    logging.info("✅ Preprocessing complete.")
    return df, train, val, test

if __name__ == "__main__":
    run_preprocessing()
