# 🆘 Disaster Relief Resource Matching Agent

An AI-assisted agent that matches disaster-relief **needs** (food, medicine,
shelter, water, transport) with available **resources**, scores urgency, and
prepares a clear coordination message for a **human coordinator to review
and approve**. The agent never auto-dispatches resources.

## Problem Statement

During emergencies, people need food, medicine, shelter, or transport while
volunteers/NGOs have resources sitting idle or scattered. This project
builds a working pipeline that takes need/resource data, matches them,
scores urgency, and produces an actionable, human-readable output — while
keeping a human in the loop for every real dispatch decision.

## Architecture

A three-stage AI pipeline sits behind the matching/guardrail logic:

```
Need submitted
   │
   ├─► Groq fast-triage (independent urgency/category read, low-latency,
   │    runs on every submission — src/groq_triage.py)
   │
   ├─► Trained urgency classifier (TF-IDF + Logistic Regression on the
   │    Kaggle Disaster Tweets dataset — src/train_classifier.py)
   │
   ├─► Rule-based matching engine (location + category, ranked by
   │    coverage — src/matching.py)
   │
   ├─► Guardrails (validation, escalation, human-review flags —
   │    src/guardrails.py)
   │
   ├─► RAG retrieval (live ReliefWeb API reports + TF-IDF similarity
   │    search over historical disaster tweets — src/rag.py, src/reliefweb.py)
   │
   ├─► Claude-generated coordination message, grounded in the RAG
   │    context and the Groq triage read (src/ai_message.py)
   │
   └─► Human approval → SQLite audit log (src/memory.py)
```

Every stage fails soft: no API key, a network error, or a missing model
file all fall back to a deterministic rule/template so the pipeline (and
any live demo) never breaks.

## Dataset / Reference Source

- Sample synthetic data: `data/needs.csv`, `data/resources.csv` (15 rows
  each, modeled on Delhi/Mumbai/Chennai/Bengaluru/Kolkata/Hyderabad/Pune
  relief scenarios)
- Reference dataset: [Disaster Tweets (Kaggle)](https://www.kaggle.com/datasets/vstepanenko/disaster-tweets)
  — `data/disaster_tweets.csv` (11,370 labeled tweets, `target=1` for
  genuine disaster/emergency text). Used both to (a) **train** the urgency
  classifier (`src/train_classifier.py`) and (b) build a TF-IDF similarity
  index for RAG historical examples (`src/rag.py`).
- Live source: the [ReliefWeb API](https://reliefweb.int/help/api) (UN
  OCHA) — free, public, no key required — for real-time humanitarian
  situation reports (`src/reliefweb.py`).

> `needs.csv` / `resources.csv` are synthetic starter data, clearly labeled
> as such. `disaster_tweets.csv` is the real Kaggle dataset.

## Tools Used

Python, pandas, scikit-learn, Anthropic Claude API, Groq API, ReliefWeb
API, Streamlit, pydeck, SQLite

## AI / Agent Components

- **Groq fast-triage** (`src/groq_triage.py`): an independent, low-latency
  urgency/category read on every submission, used as a secondary
  cross-check signal for Claude's message — not a re-scoring of the need.
  Falls back to a keyword heuristic if Groq is unavailable.
- **Trained urgency classifier** (`src/train_classifier.py`): TF-IDF +
  Logistic Regression trained on 11,370 Kaggle disaster tweets — **85%
  test accuracy, 0.66 F1 on the minority "disaster" class**. Boosts the
  manually reported urgency score (`src/urgency.py`).
- **RAG retrieval** (`src/rag.py`): combines live ReliefWeb reports for the
  need's location/category with the most textually-similar historical
  disaster tweets (TF-IDF + cosine similarity), grounding Claude's message
  in real context rather than letting it invent details.
- **Matching engine** (`src/matching.py`): rule-based, location + category
  match, ranked by coverage (`resource.quantity / need.quantity`).
- **Guardrails** (`src/guardrails.py`): rejects invalid input, flags
  low-confidence/unavailable matches, escalates high-urgency unmatched
  needs — every match still requires human sign-off.
- **Claude message generation** (`src/ai_message.py`): turns the
  structured match + RAG context + Groq triage read into a short, factual
  coordination message for a human reviewer.
- **Live map simulation** (`app/app.py`, Tab 4): a pydeck map of India
  (`ScatterplotLayer` for resources/needs, `ArcLayer` for matches)
  streaming a simulated incoming disaster feed through the full pipeline
  in real time, ending on a "spotlight" case that runs the complete
  Groq → RAG → Claude sequence end-to-end.

## How to Run

```bash
pip install -r requirements.txt

# Train the urgency classifier on the Kaggle disaster-tweets dataset
# (only needed once; saved to data/urgency_model.joblib)
python -m src.train_classifier

# Copy the example env file and add your keys
cp .env.example .env
# then edit .env: ANTHROPIC_API_KEY=..., GROQ_API_KEY=..., RELIEFWEB_APPNAME=...
# (the app runs with template/fallback behavior even without keys set)

# CLI demo (batch run over all needs.csv rows)
python -m src.main

# Interactive prototype (Submit / Batch Run / Logs / Live Map tabs)
streamlit run app/app.py
```

## Repository Structure

```
.
├── data/
│   ├── needs.csv
│   ├── resources.csv
│   ├── disaster_tweets.csv     # Kaggle dataset (11,370 labeled tweets)
│   ├── urgency_model.joblib    # trained classifier (gitignored, generated locally)
│   └── agent_logs.db           # gitignored, created on first run
├── src/
│   ├── preprocessing.py     # data loading, cleaning, classifier-based text boost
│   ├── train_classifier.py  # trains the urgency/disaster-relevance classifier
│   ├── urgency.py           # urgency scoring & ranking
│   ├── matching.py          # rule-based matching engine
│   ├── guardrails.py        # validation, escalation, human-in-loop rules
│   ├── groq_triage.py       # fast independent triage via Groq
│   ├── rag.py                # RAG retrieval: live ReliefWeb + historical tweets
│   ├── reliefweb.py         # ReliefWeb API connector
│   ├── geo.py                # offline city -> lat/lon lookup for the map
│   ├── ai_message.py        # Claude coordination-message generation
│   ├── memory.py            # SQLite logging
│   └── main.py               # agent controller / CLI entry point
├── app/
│   └── app.py                # Streamlit prototype (4 tabs: submit, batch, logs, live map)
├── requirements.txt
├── .env.example
└── README.md
```

## Results / Sample Output

Running `python -m src.main` on the sample data matches 14/15 needs (93%
coverage), correctly flags one Chennai medicine request as
`RESOURCE_UNAVAILABLE` (best candidate marked "busy"), and ranks needs by
computed urgency before processing.

## Limitations

- Sample `needs.csv`/`resources.csv` are synthetic, not live data
- The urgency classifier is a TF-IDF + Logistic Regression baseline (85%
  accuracy, weaker recall on the minority "disaster" class) trained on
  generic tweets, not relief-specific need descriptions
- Matching is location + category exact-match only; no fuzzy/geo-distance
  matching yet
- Claude/Groq/ReliefWeb calls depend on API availability; each falls back
  to a deterministic template/heuristic if a key is missing or a call fails
- This is a decision-support tool only — it never auto-dispatches resources

## Responsible Use

All matches require explicit human approval before any real-world action.
The agent is designed to reduce coordination effort and surface urgent
unmatched needs — not to replace human judgment in a safety-critical
domain.

## Future Improvements

- Add embeddings-based semantic matching for near-miss categories/locations
- Add real geo-distance-aware matching (beyond exact city match)
- Persist approvals/rejections back into the log for a feedback loop

## Team

Team Transformers

- Hiten Gogia
- Rishita Jain
- Aditya Garg