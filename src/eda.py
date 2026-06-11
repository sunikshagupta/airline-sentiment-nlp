"""
eda.py
======
Step 2 of the NLP pipeline.
Generates all EDA charts and saves them to outputs/.

Charts produced:
  1. Sentiment class distribution (bar chart)
  2. Sentiment by airline (grouped bar)
  3. Top negative reasons (horizontal bar)
  4. Tweet length distribution by sentiment (boxplot)
  5. Word clouds — one per sentiment class
  6. Confidence score distributions

Run:
    python src/eda.py
"""

import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from wordcloud import WordCloud

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s")

PALETTE = {
    "negative": "#E24B4A",
    "neutral" : "#888780",
    "positive": "#1D9E75",
}

# ── helpers ──────────────────────────────────────────────────────────────────

def save(fig, name: str):
    path = Path("outputs") / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved → {path}")

# ── 1. Sentiment class distribution ──────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame):
    counts = df["airline_sentiment"].value_counts().reindex(["negative","neutral","positive"])
    total  = len(df)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values,
                  color=[PALETTE[s] for s in counts.index],
                  edgecolor="white", linewidth=0.8, width=0.55)

    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 80,
                f"{count:,}\n({count/total*100:.1f}%)",
                ha="center", va="bottom", fontsize=10)

    ax.set_title("Sentiment Class Distribution\n(Twitter US Airline Dataset)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Sentiment", fontsize=11)
    ax.set_ylabel("Number of Tweets", fontsize=11)
    ax.set_ylim(0, counts.max() * 1.18)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines[["top","right"]].set_visible(False)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    # annotation
    ax.annotate("⚠ Imbalanced: 63% negative",
                xy=(0.97, 0.95), xycoords="axes fraction",
                ha="right", fontsize=9, color="#E24B4A",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF0F0", edgecolor="#E24B4A"))

    save(fig, "01_class_distribution.png")

# ── 2. Sentiment by airline ───────────────────────────────────────────────────

def plot_sentiment_by_airline(df: pd.DataFrame):
    pivot = (df.groupby(["airline","airline_sentiment"])
               .size()
               .unstack(fill_value=0)
               .reindex(columns=["negative","neutral","positive"]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: raw counts
    pivot.plot(kind="bar", ax=axes[0],
               color=[PALETTE[s] for s in ["negative","neutral","positive"]],
               edgecolor="white", linewidth=0.5, width=0.7)
    axes[0].set_title("Tweet Count by Airline & Sentiment", fontsize=12, fontweight="bold")
    axes[0].set_xlabel(""); axes[0].set_ylabel("Tweet Count")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend(title="Sentiment")
    axes[0].spines[["top","right"]].set_visible(False)

    # Right: percentage stacked
    pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    pct.plot(kind="bar", stacked=True, ax=axes[1],
             color=[PALETTE[s] for s in ["negative","neutral","positive"]],
             edgecolor="white", linewidth=0.5, width=0.7)
    axes[1].set_title("Sentiment Share by Airline (%)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel(""); axes[1].set_ylabel("Percentage")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend(title="Sentiment")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    axes[1].spines[["top","right"]].set_visible(False)

    fig.suptitle("Key Insight: United & US Airways dominate negative volume;\nVirgin America most balanced",
                 fontsize=10, color="#555", y=1.01)
    fig.tight_layout()
    save(fig, "02_sentiment_by_airline.png")

# ── 3. Top negative reasons ───────────────────────────────────────────────────

def plot_negative_reasons(df: pd.DataFrame):
    reasons = (df[df["airline_sentiment"] == "negative"]["negativereason"]
                 .dropna()
                 .value_counts()
                 .head(10))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(reasons.index[::-1], reasons.values[::-1],
                   color="#E24B4A", edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, reasons.values[::-1]):
        ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9)

    ax.set_title("Top 10 Reasons for Negative Tweets", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Tweets"); ax.set_ylabel("")
    ax.spines[["top","right"]].set_visible(False)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    save(fig, "03_negative_reasons.png")

# ── 4. Tweet length distribution ─────────────────────────────────────────────

def plot_tweet_length(df: pd.DataFrame):
    df = df.copy()
    df["word_count"] = df["clean_text"].str.split().str.len()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Boxplot
    order = ["negative","neutral","positive"]
    data_by_sentiment = [df[df["airline_sentiment"] == s]["word_count"] for s in order]
    bp = axes[0].boxplot(data_by_sentiment, labels=order, patch_artist=True,
                         medianprops=dict(color="white", linewidth=2))
    for patch, s in zip(bp["boxes"], order):
        patch.set_facecolor(PALETTE[s])
    axes[0].set_title("Word Count Distribution by Sentiment", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Sentiment"); axes[0].set_ylabel("Word Count")
    axes[0].spines[["top","right"]].set_visible(False)

    # KDE
    for sentiment in order:
        subset = df[df["airline_sentiment"] == sentiment]["word_count"]
        subset.plot.kde(ax=axes[1], color=PALETTE[sentiment], linewidth=2, label=sentiment)
    axes[1].set_title("Word Count Density by Sentiment", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Word Count"); axes[1].set_ylabel("Density")
    axes[1].legend(title="Sentiment")
    axes[1].spines[["top","right"]].set_visible(False)
    axes[1].set_xlim(0, 40)

    fig.tight_layout()
    save(fig, "04_tweet_length.png")

# ── 5. Word clouds ────────────────────────────────────────────────────────────

def plot_wordclouds(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sentiments = ["negative", "neutral", "positive"]

    for ax, sentiment in zip(axes, sentiments):
        text = " ".join(df[df["airline_sentiment"] == sentiment]["clean_text"].dropna())
        wc = WordCloud(
            width=600, height=400,
            background_color="white",
            colormap="Reds" if sentiment == "negative" else
                     "Greens" if sentiment == "positive" else "Greys",
            max_words=80,
            collocations=False,
        ).generate(text)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"{sentiment.capitalize()} Tweets",
                     fontsize=13, fontweight="bold", color=PALETTE[sentiment])

    fig.suptitle("Most Frequent Words per Sentiment Class", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "05_wordclouds.png")

# ── 6. Confidence scores ──────────────────────────────────────────────────────

def plot_confidence(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    for sentiment in ["negative","neutral","positive"]:
        subset = df[df["airline_sentiment"] == sentiment]["airline_sentiment_confidence"].dropna()
        subset.plot.kde(ax=ax, color=PALETTE[sentiment], linewidth=2, label=sentiment)

    ax.set_title("Labeller Confidence Score Distribution by Sentiment",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Confidence Score"); ax.set_ylabel("Density")
    ax.set_xlim(0, 1.05)
    ax.legend(title="Sentiment")
    ax.spines[["top","right"]].set_visible(False)
    ax.annotate("Low confidence labels\nmay be noisy",
                xy=(0.45, 0.5), fontsize=9, color="#888",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5", edgecolor="#CCC"))
    fig.tight_layout()
    save(fig, "06_confidence_scores.png")

# ── Business insights summary ─────────────────────────────────────────────────

def print_insights(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("KEY BUSINESS INSIGHTS")
    print("=" * 60)

    total = len(df)
    neg   = (df["airline_sentiment"] == "negative").sum()
    print(f"\n1. {neg/total*100:.0f}% of all airline tweets are negative.")
    print("   This is not a representative sample — Twitter skews towards complaints.\n")

    worst = df[df["airline_sentiment"]=="negative"]["airline"].value_counts().idxmax()
    print(f"2. {worst} receives the most negative tweets in raw volume.")

    vamask = df["airline"] == "Virgin America"
    va_neg_pct = (df[vamask]["airline_sentiment"]=="negative").mean()*100
    print(f"3. Virgin America has the lowest negative rate at {va_neg_pct:.0f}%.\n")

    top_reason = df["negativereason"].value_counts().index[0]
    top_count  = df["negativereason"].value_counts().iloc[0]
    print(f"4. Top complaint: '{top_reason}' — {top_count:,} tweets.\n")

    avg_neg_words = df[df["airline_sentiment"]=="negative"]["clean_text"].str.split().str.len().mean()
    avg_pos_words = df[df["airline_sentiment"]=="positive"]["clean_text"].str.split().str.len().mean()
    print(f"5. Negative tweets average {avg_neg_words:.1f} words vs {avg_pos_words:.1f} for positive.")
    print("   Complaints tend to be more detailed than compliments.\n")

# ── Main ─────────────────────────────────────────────────────────────────────

def run_eda():
    Path("outputs").mkdir(exist_ok=True)
    df = pd.read_csv("data/tweets_clean.csv")
    logging.info(f"Loaded {len(df):,} rows for EDA")

    logging.info("Generating chart 1/6 — class distribution")
    plot_class_distribution(df)

    logging.info("Generating chart 2/6 — sentiment by airline")
    plot_sentiment_by_airline(df)

    logging.info("Generating chart 3/6 — negative reasons")
    plot_negative_reasons(df)

    logging.info("Generating chart 4/6 — tweet length")
    plot_tweet_length(df)

    logging.info("Generating chart 5/6 — word clouds")
    plot_wordclouds(df)

    logging.info("Generating chart 6/6 — confidence scores")
    plot_confidence(df)

    print_insights(df)
    logging.info("✅ EDA complete. 6 charts saved to outputs/")

if __name__ == "__main__":
    run_eda()
