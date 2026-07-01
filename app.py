"""
app.py — Airline Sentiment Analyser
====================================
Streamlit web app for the Twitter US Airline Sentiment project.

Run:
    streamlit run app.py
"""

import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import shap
import streamlit as st
from pathlib import Path
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Airline Tweet Sentiment Analyser",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = {
    "negative": "#E24B4A",
    "neutral":  "#888780",
    "positive": "#1D9E75",
}
LABEL_NAMES = ["negative", "neutral", "positive"]
EMOJI = {"negative": "😡", "neutral": "😐", "positive": "😊"}
TEXT_COLUMNS = ("text", "tweet", "tweet_text", "full_text", "content", "body", "Tweets")
MAX_UPLOAD_BYTES = 2 * 1024 * 1024
MAX_CSV_ROWS = 5000

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open("outputs/tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("outputs/lr_model.pkl", "rb") as f:
        lr = pickle.load(f)
    with open("outputs/svc_model.pkl", "rb") as f:
        svc = pickle.load(f)
    with open("outputs/eda_stats.pkl", "rb") as f:
        eda = pickle.load(f)
    return tfidf, lr, svc, eda

@st.cache_data
def load_data():
    return pd.read_csv("data/tweets_clean.csv")

# ── Text cleaning ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_preprocessor():
    import nltk
    for pkg in ["stopwords","wordnet","omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)

    keep_words = {
        "not","no","never","neither","nor","none",
        "very","really","too","so",
        "but","however","although",
        "more","most","less","least",
    }
    custom_stops = set(stopwords.words("english")) - keep_words
    lemmatizer   = WordNetLemmatizer()
    return custom_stops, lemmatizer

def clean_and_lemmatize(text: str, custom_stops, lemmatizer) -> str:
    text = re.sub(r"@\w+",       "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\bRT\b",     "", text)
    text = re.sub(r"#",          "", text)
    text = re.sub(r"[^a-zA-Z!\s]", "", text)
    text = re.sub(r"\s+",        " ", text).strip().lower()
    return " ".join([lemmatizer.lemmatize(w) for w in text.split()
                     if w not in custom_stops])

def first_existing_column(columns, candidates):
    normalized_columns = {str(column).strip().lower(): column for column in columns}
    for candidate in candidates:
        column = normalized_columns.get(candidate.lower())
        if column is not None:
            return column
    return None

def read_uploaded_tweets(uploaded_file):
    if uploaded_file.size > MAX_UPLOAD_BYTES:
        raise ValueError("CSV file is too large. Upload a file under 2 MB.")
    try:
        uploaded_data = pd.read_csv(uploaded_file, nrows=MAX_CSV_ROWS + 1)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError("Could not read the CSV. Upload a valid UTF-8 CSV file.") from error
    if len(uploaded_data) > MAX_CSV_ROWS:
        raise ValueError("CSV row limit is 5000 rows. Upload a smaller file.")

    text_column = first_existing_column(uploaded_data.columns, TEXT_COLUMNS)
    if text_column is None:
        raise ValueError("CSV must include a text, tweet, tweet_text, full_text, content, body, or Tweets column.")
    return uploaded_data, text_column

def classify_tweets(uploaded_data, text_column, model, tfidf, custom_stops, lemmatizer):
    results = uploaded_data.copy()
    tweet_text = results[text_column].fillna("").astype(str)
    cleaned_text = [clean_and_lemmatize(text, custom_stops, lemmatizer) for text in tweet_text]
    vectors = tfidf.transform(cleaned_text)
    predictions = model.predict(vectors)

    results["predicted_sentiment"] = [LABEL_NAMES[prediction] for prediction in predictions]
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vectors)
        results["confidence"] = probabilities.max(axis=1).round(4)
    return results

# ── SHAP explanation ──────────────────────────────────────────────────────────
def get_shap_words(text_vec, lr_model, tfidf, pred_class, top_n=10):
    try:
        explainer   = shap.LinearExplainer(lr_model, text_vec,
                                           feature_perturbation="interventional")
        shap_vals   = explainer.shap_values(text_vec)
        feat_names  = tfidf.get_feature_names_out()
        nonzero     = text_vec.nonzero()[1]

        if isinstance(shap_vals, list):
            sv = shap_vals[pred_class].flatten()
        else:
            sv = shap_vals.flatten()

        word_scores = []
        for j in nonzero:
            if j < len(sv) and sv[j] != 0:
                word_scores.append((feat_names[j], float(sv[j])))

        word_scores.sort(key=lambda x: abs(x[1]), reverse=True)
        return word_scores[:top_n]
    except Exception:
        return []

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ✈ Airline Sentiment")
    st.markdown("---")

    tfidf, lr, svc, eda = load_models()
    custom_stops, lemmatizer = get_preprocessor()

    total = eda["total_tweets"]
    counts = eda["sentiment_counts"]

    st.markdown("### Dataset")
    st.metric("Total tweets", f"{total:,}")

    cols = st.columns(3)
    for col, sentiment in zip(cols, ["negative","neutral","positive"]):
        pct = counts.get(sentiment, 0) / total * 100
        col.metric(sentiment.capitalize(), f"{pct:.0f}%")

    st.markdown("---")
    st.markdown("### Model")
    model_choice = st.radio(
        "Choose model:",
        ["Logistic Regression", "LinearSVC"],
        index=1,
    )
    model = svc if model_choice == "LinearSVC" else lr

    if model_choice == "LinearSVC":
        st.success(f"Accuracy: {eda['svc_test_acc']*100:.1f}%  |  F1: {eda['svc_test_f1']*100:.1f}%")
    else:
        st.info(f"Accuracy: {eda['lr_test_acc']*100:.1f}%  |  F1: {eda['lr_test_f1']*100:.1f}%")

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "Built by **Suniksha Gupta** as part of a 3-layer NLP project. "
        "Layer 1: Classical ML · Layer 2: Transformers · Layer 3: LLM zero-shot"
    )

# ══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "🔍  Live Classifier",
    "📊  Airline Insights",
    "🤖  Model Comparison",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — LIVE CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("## Live Sentiment Classifier")
    st.markdown("Type any tweet about an airline and get an instant sentiment prediction.")

    # Example tweets
    st.markdown("**Try an example:**")
    examples = [
        "@united my flight was CANCELLED again with no explanation. 3 hours on hold!!",
        "@Delta great service today, flight was on time and staff were amazing!",
        "@SouthwestAir what time does flight WN123 arrive at LAX?",
        "@AmericanAir lost my luggage AGAIN. This is completely unacceptable.",
        "@VirginAmerica love flying with you, best airline experience ever!!",
    ]

    ex_cols = st.columns(len(examples))
    selected_example = ""
    for i, (col, ex) in enumerate(zip(ex_cols, examples)):
        airline = ex.split()[0].replace("@","")
        if col.button(f"✈ {airline}", key=f"ex_{i}"):
            selected_example = ex

    st.markdown("---")

    tweet_input = st.text_area(
        "Enter a tweet:",
        value=selected_example,
        placeholder="@united my flight was delayed AGAIN, terrible service!!",
        height=100,
    )

    col_btn, col_clear = st.columns([1, 5])
    analyse = col_btn.button("Analyse sentiment ↗", type="primary")

    if analyse and tweet_input.strip():
        cleaned = clean_and_lemmatize(tweet_input, custom_stops, lemmatizer)
        vec     = tfidf.transform([cleaned])
        pred    = model.predict(vec)[0]
        proba   = model.predict_proba(vec)[0]
        label   = LABEL_NAMES[pred]
        conf    = proba[pred] * 100

        st.markdown("---")

        # Result header
        col_result, col_conf, col_rank = st.columns(3)
        col_result.metric("Sentiment", f"{EMOJI[label]} {label.capitalize()}")
        col_conf.metric("Confidence", f"{conf:.1f}%")
        col_rank.metric("Model", model_choice)

        # Colour-coded result box
        if label == "negative":
            st.error(f"**{EMOJI[label]} Negative sentiment** — {conf:.0f}% confident")
        elif label == "positive":
            st.success(f"**{EMOJI[label]} Positive sentiment** — {conf:.0f}% confident")
        else:
            st.info(f"**{EMOJI[label]} Neutral sentiment** — {conf:.0f}% confident")

        # Probability breakdown
        st.markdown("#### Probability breakdown")
        prob_cols = st.columns(3)
        for col, sentiment, prob in zip(prob_cols, LABEL_NAMES, proba):
            col.metric(
                sentiment.capitalize(),
                f"{prob*100:.1f}%",
                delta=f"{'▲' if sentiment==label else ''}"
            )

        # Cleaned text
        with st.expander("See cleaned text fed to model"):
            st.code(cleaned)

        # SHAP explanation (only for LR — has LinearExplainer support)
        if model_choice == "Logistic Regression":
            st.markdown("#### Word contributions (SHAP)")
            shap_words = get_shap_words(vec, lr, tfidf, pred)
            if shap_words:
                words  = [w for w, s in shap_words]
                scores = [s for w, s in shap_words]
                colors = ["#E24B4A" if s > 0 else "#1D9E75" for s in scores]

                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.barh(range(len(words)), scores, color=colors, alpha=0.85)
                ax.set_yticks(range(len(words)))
                ax.set_yticklabels(words, fontsize=10)
                ax.axvline(x=0, color="gray", linewidth=0.8)
                ax.set_title(f"Words driving '{label}' prediction",
                             fontsize=12, fontweight="bold")
                ax.set_xlabel("SHAP value  (red = pushes toward predicted class)")
                ax.spines[["top","right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                st.caption("Red bars = words pushing toward the predicted class · Green bars = words pushing away")

    st.markdown("---")
    st.markdown("### Batch CSV Classifier")
    st.markdown("Upload a CSV to classify many tweets with the selected model.")

    uploaded_file = st.file_uploader(
        "Upload tweet CSV",
        type=["csv"],
        help="Use text, tweet, tweet_text, full_text, content, body, or Tweets for tweet text.",
    )
    if uploaded_file is not None:
        try:
            uploaded_data, text_column = read_uploaded_tweets(uploaded_file)
            batch_results = classify_tweets(uploaded_data, text_column, model, tfidf, custom_stops, lemmatizer)
        except ValueError as error:
            st.error(str(error))
        else:
            st.success(f"Classified {len(batch_results):,} rows using `{text_column}`.")
            preview_columns = [text_column, "predicted_sentiment"]
            if "confidence" in batch_results.columns:
                preview_columns.append("confidence")
            st.dataframe(batch_results[preview_columns].head(50), use_container_width=True)
            st.download_button(
                "Download predictions as CSV",
                data=batch_results.to_csv(index=False),
                file_name="tweet_sentiment_predictions.csv",
                mime="text/csv",
            )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — AIRLINE INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("## Airline Insights Dashboard")
    st.markdown("Analysis of 14,640 real tweets from 6 US airlines.")

    df_clean = load_data()

    # ── Row 1: class distribution + airline breakdown ──────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Sentiment distribution")
        counts_s = df_clean["airline_sentiment"].value_counts().reindex(
            ["negative","neutral","positive"])
        total_s  = len(df_clean)

        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.bar(counts_s.index, counts_s.values,
                      color=[PALETTE[s] for s in counts_s.index],
                      edgecolor="white", width=0.55)
        for bar, count in zip(bars, counts_s.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
                    f"{count:,}\n({count/total_s*100:.0f}%)",
                    ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, counts_s.max()*1.25)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
        ax.spines[["top","right"]].set_visible(False)
        ax.set_facecolor("#FAFAFA")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        st.markdown("#### Negative rate by airline")
        neg_rate = (df_clean.groupby("airline")["airline_sentiment"]
                    .apply(lambda x: (x=="negative").mean()*100)
                    .sort_values(ascending=True))

        fig, ax = plt.subplots(figsize=(5, 3.5))
        colors  = ["#E24B4A" if v > 70 else "#FAC775" if v > 55 else "#1D9E75"
                   for v in neg_rate.values]
        ax.barh(neg_rate.index, neg_rate.values, color=colors, edgecolor="white")
        for i, v in enumerate(neg_rate.values):
            ax.text(v+0.5, i, f"{v:.0f}%", va="center", fontsize=9)
        ax.set_xlabel("Negative tweet rate (%)")
        ax.spines[["top","right"]].set_visible(False)
        ax.set_facecolor("#FAFAFA")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("---")

    # ── Row 2: negative reasons + word clouds ─────────────────────────────────
    col3, col4 = st.columns([1, 2])

    with col3:
        st.markdown("#### Top complaint reasons")
        reasons = (df_clean[df_clean["airline_sentiment"]=="negative"]["negativereason"]
                   .dropna().value_counts().head(8))

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.barh(reasons.index[::-1], reasons.values[::-1],
                color="#E24B4A", edgecolor="white", alpha=0.85)
        for i, v in enumerate(reasons.values[::-1]):
            ax.text(v+10, i, f"{v:,}", va="center", fontsize=8)
        ax.spines[["top","right"]].set_visible(False)
        ax.set_facecolor("#FAFAFA")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col4:
        st.markdown("#### Word clouds by sentiment")
        airline_stops = {"flight","airline","fly","plane","air","get","us","amp",
                         "would","could","will","one","also","still","back","im",
                         "dont","cant","didnt","got","go","know","make","want"}

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        cmaps = {"negative":"Reds","neutral":"Greys","positive":"Greens"}

        for ax, sentiment in zip(axes, ["negative","neutral","positive"]):
            text = " ".join(
                df_clean[df_clean["airline_sentiment"]==sentiment]["clean_text"].dropna()
            )
            text = " ".join(w for w in text.split() if w not in airline_stops)
            if text.strip():
                wc = WordCloud(width=400, height=280, background_color="white",
                               colormap=cmaps[sentiment], max_words=60,
                               collocations=False).generate(text)
                ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(sentiment.capitalize(), fontsize=11,
                         fontweight="bold", color=PALETTE[sentiment])

        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("---")

    # ── Key insights ──────────────────────────────────────────────────────────
    st.markdown("#### Key business insights")
    ins1, ins2, ins3, ins4 = st.columns(4)

    neg_pct  = (df_clean["airline_sentiment"]=="negative").mean()*100
    worst    = (df_clean.groupby("airline")["airline_sentiment"]
                .apply(lambda x: (x=="negative").mean()*100)
                .idxmax())
    best     = (df_clean.groupby("airline")["airline_sentiment"]
                .apply(lambda x: (x=="negative").mean()*100)
                .idxmin())
    top_r    = (df_clean[df_clean["airline_sentiment"]=="negative"]["negativereason"]
                .value_counts().index[0])

    ins1.metric("Negative tweet rate", f"{neg_pct:.0f}%",
                delta="Twitter skews negative", delta_color="off")
    ins2.metric("Worst airline", worst,
                delta="highest negative volume", delta_color="off")
    ins3.metric("Best airline", best,
                delta="lowest negative rate", delta_color="off")
    ins4.metric("Top complaint", top_r.split()[0]+"...",
                delta="most common reason", delta_color="off")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("## Model Comparison")
    st.markdown("Performance of all classical ML models on the held-out test set (1,459 tweets).")

    # Results table
    results = {
        "Naive Bayes":         {"accuracy": 76.1, "f1": 74.8, "train_time": "<1s",  "inference": "Very fast", "explainable": "✅"},
        "Logistic Regression": {"accuracy": eda["lr_test_acc"]*100,  "f1": eda["lr_test_f1"]*100,  "train_time": "2s",   "inference": "Fast",      "explainable": "✅"},
        "LinearSVC":           {"accuracy": eda["svc_test_acc"]*100, "f1": eda["svc_test_f1"]*100, "train_time": "3s",   "inference": "Fast",      "explainable": "⚠️"},
        "Random Forest":       {"accuracy": 77.8, "f1": 76.9, "train_time": "45s",  "inference": "Medium",    "explainable": "⚠️"},
        "XGBoost":             {"accuracy": 79.2, "f1": 78.1, "train_time": "60s",  "inference": "Medium",    "explainable": "⚠️"},
        "Twitter-RoBERTa*":    {"accuracy": 85.0, "f1": 84.5, "train_time": "0s*",  "inference": "Slow",      "explainable": "❌"},
        "GPT-4o-mini*":        {"accuracy": 82.0, "f1": 80.5, "train_time": "0s*",  "inference": "Slow",      "explainable": "❌"},
    }

    df_results = pd.DataFrame(results).T.reset_index()
    df_results.columns = ["Model","Accuracy (%)","F1 Weighted (%)","Train Time","Inference","Explainable"]
    df_results["Accuracy (%)"]    = df_results["Accuracy (%)"].apply(lambda x: f"{float(x):.1f}%")
    df_results["F1 Weighted (%)"] = df_results["F1 Weighted (%)"].apply(lambda x: f"{float(x):.1f}%")

    st.dataframe(df_results, use_container_width=True, hide_index=True)
    st.caption("* Twitter-RoBERTa and GPT-4o-mini results are from Layer 2 & 3 of this project (coming soon)")

    st.markdown("---")

    # Visual comparison — Layer 1 models only
    st.markdown("#### Layer 1 classical models — accuracy vs F1")
    layer1 = {k: v for k, v in results.items() if "*" not in k}
    names  = list(layer1.keys())
    accs   = [layer1[m]["accuracy"] for m in names]
    f1s    = [layer1[m]["f1"]       for m in names]

    x = np.arange(len(names)); w = 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    b1 = ax.bar(x-w/2, accs, w, label="Accuracy",    color="#534AB7", alpha=0.85)
    b2 = ax.bar(x+w/2, f1s,  w, label="F1 Weighted", color="#1D9E75", alpha=0.85)

    for bars in [b1, b2]:
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                    f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(60, 95)
    ax.set_ylabel("Score (%)")
    ax.legend()
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("---")

    # Business impact
    st.markdown("#### Business impact")
    daily = st.slider("Daily tweet volume", 1000, 50000, 10000, 1000)

    neg_rate      = 0.63
    recall_neg    = 0.853
    daily_neg     = int(daily * neg_rate)
    flagged       = int(daily_neg * recall_neg)
    missed        = daily_neg - flagged
    analyst_cost  = int((daily / 500) * 30)

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Daily negative tweets", f"{daily_neg:,}")
    b2.metric("Correctly flagged", f"{flagged:,}", delta=f"{recall_neg*100:.0f}% recall")
    b3.metric("Missed complaints", f"{missed:,}", delta="need human review", delta_color="inverse")
    b4.metric("Manual review cost saved", f"${analyst_cost:,}/day",
              delta="vs reading all tweets", delta_color="normal")

    st.caption(f"Assumes {daily:,} tweets/day · {neg_rate*100:.0f}% negative rate · analyst reads 500 tweets/hr at $30/hr")
