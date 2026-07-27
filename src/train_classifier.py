"""
train_classifier.py
--------------------
Trains a lightweight text classifier on the Kaggle "Disaster Tweets" dataset
(https://www.kaggle.com/datasets/vstepanenko/disaster-tweets) to predict
whether a piece of text describes a genuine disaster/emergency situation.

This model is the real AI/ML component behind urgency scoring: instead of a
hand-written keyword list, `need.description` text is run through a trained
TF-IDF + Logistic Regression classifier, and the predicted probability is
used to boost the reported urgency score.

Run once to (re)build the model:
    python -m src.train_classifier

Produces: data/urgency_model.joblib
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import joblib

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "disaster_tweets.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "urgency_model.joblib")


def train_and_save(data_path: str = DATA_PATH, model_path: str = MODEL_PATH):
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["text", "target"])

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["target"], test_size=0.2, random_state=42, stratify=df["target"]
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.3f}")
    print(classification_report(y_test, preds, target_names=["not_disaster", "disaster"]))

    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")
    return pipeline, acc


if __name__ == "__main__":
    train_and_save()
