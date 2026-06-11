# ✈ Airline Tweet Sentiment Analysis
### End-to-end NLP project — Classical ML · Feature Engineering · Explainability · Streamlit

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://airline-sentiment-nlp-5jyhu9cmpfy9hketmnqioa.streamlit.app)

---

## 🔴 Live Demo
**[Try the app →](https://airline-sentiment-nlp-5jyhu9cmpfy9hketmnqioa.streamlit.app)**

Type any airline tweet and get an instant sentiment prediction with confidence score and word-level explanations.

---

## 📌 Project Overview

Built a 3-layer NLP pipeline to classify 14,640 real tweets about US airlines as **positive**, **negative**, or **neutral**.

| Layer | Approach | Accuracy |
|-------|----------|----------|
| Layer 1 | Classical ML (TF-IDF + 5 models) | ~80% |
| Layer 2 | Transformer (Twitter-RoBERTa) | ~85% *(coming soon)* |
| Layer 3 | LLM zero-shot (GPT-4o-mini) | ~82% *(coming soon)* |

---

## 📊 Key Findings

- **63%** of airline tweets are negative — Twitter skews toward complaints
- **US Airways** has the highest negative rate at 78%
- **Virgin America** is the most balanced airline at 36% negative
- Top complaint: **Customer Service Issue** (2,910 tweets)
- Negative tweets average **18 words** vs 12.5 for positive — complaints are more detailed
- **Neutral class** is hardest to classify (~65% F1) due to vocabulary overlap with negative

---

## 🏗 Project Structure

```
airline-sentiment-nlp/
├── notebooks/
│   ├── 00_data_cleaning.ipynb        # Missing values, stopwords, lemmatization
│   ├── 01_EDA.ipynb                  # Temporal analysis, vocabulary overlap, correlation
│   ├── 02_feature_engineering.ipynb  # BoW vs TF-IDF vs Word2Vec comparison
│   └── 03_modelling.ipynb            # 5 models, CV, grid search, SHAP, error analysis
├── src/
│   ├── preprocessing.py              # Full cleaning pipeline
│   ├── eda.py                        # EDA chart generation
│   └── models.py                     # Model training scripts
├── outputs/                          # Saved models, charts, evaluation results
├── app.py                            # Streamlit web application
├── setup_models.py                   # Train and save models for the app
└── requirements.txt
```

---

## 🔬 Methodology

### Data Cleaning
- Removed @mentions, URLs, RT markers, hashtag symbols
- Kept exclamation marks (sentiment signal)
- Custom stopword list preserving negation words (not, no, never)
- Lemmatization over stemming for interpretability

### Feature Engineering
Compared three approaches:

| Method | Accuracy | Dimensions | Notes |
|--------|----------|------------|-------|
| Bag of Words | 78.8% | 15,000 | Fast, ignores word importance |
| TF-IDF | 78.5% | 15,000 | Best classical approach |
| Word2Vec | — | 100 | Limited by small corpus |

### Models Trained

| Model | Test Accuracy | Test F1 |
|-------|--------------|---------|
| Naive Bayes | 76.1% | 74.8% |
| Logistic Regression | 79.2% | 79.6% |
| LinearSVC | 80.5% | 79.6% |
| Random Forest | 77.8% | 76.9% |
| XGBoost | 79.2% | 78.1% |

### Explainability
Used **SHAP (SHapley Additive exPlanations)** to identify which words drive each prediction:
- Negative drivers: *cancelled, delayed, hold, terrible, worst*
- Positive drivers: *thank, great, amazing, love, awesome*
- Neutral is hardest — vocabulary overlaps heavily with negative class

### Class Imbalance
- Dataset: 63% negative / 21% neutral / 16% positive
- Strategy: `class_weight='balanced'` in all models
- Tested SMOTE oversampling — found it hurts performance on sparse TF-IDF vectors

---

## 📈 Business Impact

At 10,000 tweets/day with 85.3% recall on negatives:
- **8,530** complaints correctly flagged per day
- **1,470** missed complaints requiring human review
- Saves ~**$600/day** vs manual review at 500 tweets/hour @ $30/hr

---

## 🚀 Run Locally

```bash
# Clone the repository
git clone https://github.com/sunikshagupta/airline-sentiment-nlp.git
cd airline-sentiment-nlp

# Create virtual environment
python -m venv nlp-env
nlp-env\Scripts\activate  # Windows
# source nlp-env/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

# Download dataset from Kaggle
# https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment
# Place Tweets.csv in data/

# Run preprocessing and train models
python src/preprocessing.py
python setup_models.py

# Launch the app
streamlit run app.py
```

---

## 🛠 Tech Stack

`Python` `Pandas` `Scikit-learn` `XGBoost` `SHAP` `NLTK` `Streamlit` `Matplotlib` `Seaborn` `WordCloud`

---

## 👩‍💻 Author

**Suniksha Gupta, PhD**
Data Scientist | NLP | Machine Learning | Python

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/suniksha-gupta)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/sunikshagupta)

---

*Part of an ongoing NLP project series. Layer 2 (Transformer) and Layer 3 (LLM zero-shot) coming soon.*
