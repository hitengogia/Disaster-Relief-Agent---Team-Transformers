"""
urgency.py
----------
Computes a final urgency score/level for a need, combining the manually
reported urgency (0-10) with an optional text-based booster derived from
disaster-tweet style keywords.
"""

from src.preprocessing import text_urgency_boost, disaster_relevance_probability


def compute_urgency_score(need: dict) -> int:
    """
    Combine reported urgency with a boost from the free-text description.
    The boost comes from a Logistic Regression classifier trained on the
    Kaggle Disaster Tweets dataset (falls back to a keyword heuristic if
    the trained model isn't available). Capped at 10.
    """
    base = int(need.get("urgency", 0))
    boost = text_urgency_boost(need.get("description", ""))
    return min(base + boost, 10)


def compute_model_confidence(need: dict):
    """Returns the classifier's raw disaster-relevance probability (0-1) for display, or None."""
    return disaster_relevance_probability(need.get("description", ""))


def urgency_level(score: int) -> str:
    if score >= 8:
        return "HIGH"
    elif score >= 5:
        return "MEDIUM"
    return "LOW"


def rank_needs_by_urgency(needs_df):
    """Return needs sorted by computed urgency score, highest first."""
    records = needs_df.to_dict(orient="records")
    for n in records:
        n["urgency_score"] = compute_urgency_score(n)
        n["urgency_level"] = urgency_level(n["urgency_score"])
    return sorted(records, key=lambda x: x["urgency_score"], reverse=True)
