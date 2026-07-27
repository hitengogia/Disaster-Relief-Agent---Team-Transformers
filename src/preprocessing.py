"""
preprocessing.py
-----------------
Loads and cleans the needs/resources CSV files, and (optionally) uses the
Kaggle "Disaster Tweets" dataset to boost urgency scores with keyword signals
extracted from real disaster-related text.

Dataset reference:
https://www.kaggle.com/datasets/vstepanenko/disaster-tweets

If the Kaggle CSV is not found, the module falls back to a small built-in
keyword list so the rest of the pipeline still works end-to-end.
"""

import os
import pandas as pd

REQUIRED_NEED_COLS = ["id", "location", "category", "quantity", "urgency"]
REQUIRED_RESOURCE_COLS = ["id", "location", "category", "quantity", "contact", "status"]

# Keywords that historically correlate with high-urgency disaster tweets.
# Used as a LAST-RESORT fallback if the trained classifier (see
# train_classifier.py) isn't available -- keeps the pipeline runnable
# even without the model file.
HIGH_URGENCY_KEYWORDS = [
    "trapped", "dying", "urgent", "emergency", "collapsed", "flooding",
    "stranded", "critical", "injured", "rescue", "sos", "help needed",
    "no food", "no water", "life threatening"
]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "urgency_model.joblib")
_MODEL_CACHE = None


def _load_model():
    """Lazily load the trained disaster-relevance classifier (cached)."""
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    if not os.path.exists(MODEL_PATH):
        _MODEL_CACHE = False
        return False
    try:
        import joblib
        _MODEL_CACHE = joblib.load(MODEL_PATH)
    except Exception:
        _MODEL_CACHE = False
    return _MODEL_CACHE


def load_csv(path: str, required_cols: list) -> pd.DataFrame:
    """Load a CSV and validate that required columns exist."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    # Basic cleaning
    df = df.dropna(subset=required_cols)
    for col in ["location", "category"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
    if "status" in df.columns:
        df["status"] = df["status"].astype(str).str.strip().str.lower()

    return df.reset_index(drop=True)


def load_needs(path: str = "data/needs.csv") -> pd.DataFrame:
    return load_csv(path, REQUIRED_NEED_COLS)


def load_resources(path: str = "data/resources.csv") -> pd.DataFrame:
    return load_csv(path, REQUIRED_RESOURCE_COLS)


def load_disaster_tweets(path: str = "data/disaster_tweets.csv"):
    """
    Optionally load the Kaggle disaster-tweets dataset for keyword/context
    reference. Returns None if the file isn't present -- this keeps the
    pipeline runnable without the external dataset.
    """
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return None


def _keyword_boost(text: str) -> int:
    """Fallback keyword-based booster (0-3), used only if the trained model is unavailable."""
    if not isinstance(text, str):
        return 0
    text_lower = text.lower()
    hits = sum(1 for kw in HIGH_URGENCY_KEYWORDS if kw in text_lower)
    return min(hits, 3)


def text_urgency_boost(text: str) -> int:
    """
    Urgency booster (0-3) for a need's free-text description.

    Primary path: a Logistic Regression classifier trained on the Kaggle
    "Disaster Tweets" dataset (src/train_classifier.py) predicts the
    probability the text describes a real disaster/emergency; higher
    probability -> higher boost.

    Fallback path: if the trained model file isn't present, falls back to a
    simple keyword heuristic so the pipeline keeps working.
    """
    if not isinstance(text, str) or not text.strip():
        return 0

    model = _load_model()
    if model:
        try:
            proba = model.predict_proba([text])[0][1]  # P(disaster-relevant)
            if proba >= 0.85:
                return 3
            elif proba >= 0.6:
                return 2
            elif proba >= 0.4:
                return 1
            return 0
        except Exception:
            pass  # fall through to keyword heuristic

    return _keyword_boost(text)


def disaster_relevance_probability(text: str):
    """
    Returns the raw model probability (0-1) that `text` describes a real
    disaster/emergency, or None if the trained model isn't available.
    Useful for displaying model confidence in the UI/logs.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    model = _load_model()
    if not model:
        return None
    try:
        return round(float(model.predict_proba([text])[0][1]), 3)
    except Exception:
        return None


if __name__ == "__main__":
    needs = load_needs("../data/needs.csv")
    resources = load_resources("../data/resources.csv")
    print(f"Loaded {len(needs)} needs and {len(resources)} resources")
