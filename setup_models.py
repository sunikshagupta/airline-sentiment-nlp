"""
setup_models.py
Run this once to train and save all models needed by the Streamlit app.

    python setup_models.py
"""
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score

print("Loading data...")
train = pd.read_csv("data/train.csv")
val   = pd.read_csv("data/val.csv")
test  = pd.read_csv("data/test.csv")
df    = pd.read_csv("data/tweets_clean.csv")

print("Loading TF-IDF vectorizer...")
with open("outputs/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

print("Transforming text...")
X_train = tfidf.transform(train["clean_text"].fillna(""))
X_test  = tfidf.transform(test["clean_text"].fillna(""))

print("Training Logistic Regression...")
lr = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0, random_state=42)
lr.fit(X_train, train["label"])

print("Training LinearSVC...")
svc = CalibratedClassifierCV(
    LinearSVC(class_weight="balanced", max_iter=2000, random_state=42)
)
svc.fit(X_train, train["label"])

print("Saving models...")
with open("outputs/lr_model.pkl", "wb") as f:
    pickle.dump(lr, f)

with open("outputs/svc_model.pkl", "wb") as f:
    pickle.dump(svc, f)

lr_acc  = accuracy_score(test["label"], lr.predict(X_test))
lr_f1   = f1_score(test["label"], lr.predict(X_test), average="weighted")
svc_acc = accuracy_score(test["label"], svc.predict(X_test))
svc_f1  = f1_score(test["label"], svc.predict(X_test), average="weighted")

eda = {
    "total_tweets"    : len(df),
    "sentiment_counts": df["airline_sentiment"].value_counts().to_dict(),
    "lr_test_acc"     : lr_acc,
    "lr_test_f1"      : lr_f1,
    "svc_test_acc"    : svc_acc,
    "svc_test_f1"     : svc_f1,
}

with open("outputs/eda_stats.pkl", "wb") as f:
    pickle.dump(eda, f)

print("Done!")
print(f"LR  : {lr_acc*100:.1f}% accuracy  {lr_f1*100:.1f}% F1")
print(f"SVC : {svc_acc*100:.1f}% accuracy  {svc_f1*100:.1f}% F1")
print("All model files saved to outputs/")
print("You can now run: streamlit run app.py")
