"""
rag.py
------
Retrieval-Augmented Generation layer: gathers grounding context for a need
before it's handed to the LLM for coordination-message writing.

Two retrieval sources are combined:
  1. LIVE  — recent ReliefWeb reports for the need's location/category
             (src/reliefweb.py). This is the actual real-time piece.
  2. STATIC — the most textually-similar rows from the Kaggle disaster-tweets
             dataset (TF-IDF + cosine similarity), giving the LLM a couple of
             real-world examples of how similar situations were described.

Important distinction (worth stating in the presentation): RAG on its own
does not make data real-time — it's just "retrieve then generate". The
live-ness here comes entirely from source #1 (ReliefWeb). Source #2 is
retrieval over a static corpus and is included because it measurably
improves grounding/specificity of the generated message, not because it's
live.

Both sources fail soft: if ReliefWeb is unreachable or the tweet corpus/
vectorizer can't be built, retrieve_context() still returns a usable
(possibly empty) context rather than raising.
"""

import os
from src.reliefweb import fetch_live_reports
from src.preprocessing import load_disaster_tweets

_VECTORIZER_CACHE = None
_TFIDF_MATRIX_CACHE = None
_TWEETS_DF_CACHE = None

DISASTER_TWEETS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "disaster_tweets.csv"
)


def _build_historical_index():
    """
    Lazily builds (and caches) a TF-IDF index over the disaster-relevant
    rows (target == 1) of the Kaggle disaster-tweets dataset. Cached at
    module level so it's only built once per process, not once per request.
    """
    global _VECTORIZER_CACHE, _TFIDF_MATRIX_CACHE, _TWEETS_DF_CACHE

    if _VECTORIZER_CACHE is not None:
        return _VECTORIZER_CACHE, _TFIDF_MATRIX_CACHE, _TWEETS_DF_CACHE

    df = load_disaster_tweets(DISASTER_TWEETS_PATH)
    if df is None or "text" not in df.columns:
        _VECTORIZER_CACHE = False
        return False, None, None

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        subset = df.copy()
        if "target" in subset.columns:
            subset = subset[subset["target"] == 1]
        subset = subset.dropna(subset=["text"]).reset_index(drop=True)

        if subset.empty:
            _VECTORIZER_CACHE = False
            return False, None, None

        vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        matrix = vectorizer.fit_transform(subset["text"])

        _VECTORIZER_CACHE = vectorizer
        _TFIDF_MATRIX_CACHE = matrix
        _TWEETS_DF_CACHE = subset
        return vectorizer, matrix, subset
    except Exception:
        _VECTORIZER_CACHE = False
        return False, None, None


def retrieve_historical_examples(description: str, top_k: int = 2) -> list:
    """
    Return up to `top_k` historically similar disaster-tweet examples
    (text only) for a given need description, using cosine similarity over
    TF-IDF vectors. Returns [] if there's no description, no model, or no
    reasonably similar match.
    """
    if not description or not description.strip():
        return []

    vectorizer, matrix, subset = _build_historical_index()
    if not vectorizer:
        return []

    try:
        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = vectorizer.transform([description])
        sims = cosine_similarity(query_vec, matrix)[0]
        top_indices = sims.argsort()[::-1][:top_k]

        examples = []
        for idx in top_indices:
            if sims[idx] <= 0.05:  # too dissimilar to be useful context
                continue
            examples.append({
                "text": subset.iloc[idx]["text"],
                "similarity": round(float(sims[idx]), 3),
            })
        return examples
    except Exception:
        return []


def retrieve_context(need: dict, top_k_live: int = 3, top_k_historical: int = 2) -> dict:
    """
    Main entry point: gathers live + historical context for a need and
    returns both the structured pieces and a compact text block ready to
    drop into an LLM prompt.

    Returns:
        {
            "live_reports": [...],
            "historical_examples": [...],
            "context_text": "..."  # human-readable, prompt-ready summary
        }
    """
    location = need.get("location", "")
    category = need.get("category", "")
    description = need.get("description", "")

    live_reports = fetch_live_reports(location, category, limit=top_k_live)
    historical_examples = retrieve_historical_examples(description, top_k=top_k_historical)

    lines = []
    if live_reports:
        lines.append("Recent live humanitarian reports (ReliefWeb):")
        for r in live_reports:
            date = f" ({r['date'][:10]})" if r.get("date") else ""
            lines.append(f"- [{r['source']}]{date} {r['title']}: {r['summary'][:200]}")
    if historical_examples:
        lines.append("Similar past disaster-related reports (reference only, not live):")
        for ex in historical_examples:
            lines.append(f"- \"{ex['text'][:200]}\"")

    context_text = "\n".join(lines) if lines else ""

    return {
        "live_reports": live_reports,
        "historical_examples": historical_examples,
        "context_text": context_text,
    }


if __name__ == "__main__":
    sample_need = {
        "location": "Mumbai",
        "category": "shelter",
        "description": "Families displaced after building collapse, urgent shelter needed",
    }
    ctx = retrieve_context(sample_need)
    print(f"Live reports: {len(ctx['live_reports'])}")
    print(f"Historical examples: {len(ctx['historical_examples'])}")
    print("\n--- context_text ---")
    print(ctx["context_text"] or "(empty)")
