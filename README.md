# 🆘 Disaster Relief Resource Matching Agent

An AI-assisted agent that matches disaster-relief **needs** (food, medicine,
shelter, water, transport) with available **resources**, scores urgency,
and prepares a clear coordination message for a **human coordinator to
review and approve**. The agent never auto-dispatches resources.

## Problem Statement

During emergencies, people need food, medicine, shelter, or transport while
volunteers/NGOs have resources sitting idle or scattered. This project
builds a working pipeline that takes need/resource data, matches them,
scores urgency, and produces an actionable, human-readable output —
while keeping a human in the loop for every real dispatch decision.

## Dataset / Reference Source

- Sample synthetic data: `data/needs.csv`, `data/resources.csv` (15 rows each,
  modeled on Delhi/Mumbai/Chennai/Bengaluru/Kolkata/Hyderabad/Pune relief scenarios)
- Reference dataset: [Disaster Tweets (Kaggle)](https://www.kaggle.com/datasets/vstepanenko/disaster-tweets)
  — `data/disaster_tweets.csv` (11,370 labeled tweets, `target=1` for genuine
  disaster/emergency text) is used to **train** a text classifier
  (`src/train_classifier.py`) that scores how urgent a need's free-text
  description sounds. If the trained model file is ever missing, the
  pipeline falls back to a simple keyword heuristic so it never breaks.

> **Note:** `needs.csv` / `resources.csv` are synthetic starter data, clearly
> labeled as such, following the project brief's fallback guidance.
> `disaster_tweets.csv` is the real Kaggle dataset.

## Tools Used

Python, pandas, Anthropic Claude API (`anthropic` SDK), Streamlit, SQLite

## Project Workflow

```
User input (need) --> Agent Controller --> Urgency Scoring --> Matching Engine
     --> Guardrail Checks --> AI Message Generation --> Human Approval --> Logs
```

1. **Input**: a need is submitted (manually via UI, or loaded in batch from `needs.csv`)
2. **Urgency scoring**: reported urgency (0-10) is combined with a keyword-based
   boost extracted from the free-text description (disaster-tweet-style signals)
3. **Matching**: rule-based engine filters resources by location + category,
   ranks by coverage (`resource.quantity / need.quantity`) and availability
4. **Guardrails**: rejects invalid input, flags unavailable/low-confidence
   matches, and escalates high-urgency needs with no match
5. **AI message generation**: Claude API turns the structured match into a
   short, factual coordination message for a human reviewer (falls back to a
   template if no API key is set, so the demo never breaks)
6. **Human-in-the-loop**: every match requires explicit approval before any
   real-world dispatch — the agent recommends, it does not act
7. **Logging**: every decision is written to a local SQLite log (`data/agent_logs.db`)
   for auditability

## AI / Agent Component

- **Trained urgency classifier** (`src/train_classifier.py`): a TF-IDF +
  Logistic Regression model trained on 11,370 labeled tweets from the Kaggle
  Disaster Tweets dataset, predicting whether a need's free-text description
  reads like a genuine emergency. Achieves **85% test accuracy** (0.66 F1 on
  the minority "disaster" class). The predicted probability boosts the
  manually-reported urgency score (`src/urgency.py`). Falls back to a keyword
  heuristic if the model file is unavailable.
- **Matching engine**: rule-based scoring (`src/matching.py`) that could be
  upgraded to embeddings-based semantic matching
- **LLM message generation** (`src/ai_message.py`): the Claude API converts
  structured JSON matches into natural-language messages a human can act on
  quickly — this is the "last mile" that makes the output usable by a
  non-technical relief coordinator, not just a developer

## How to Run

```bash
pip install -r requirements.txt

# Train the urgency classifier on the Kaggle disaster-tweets dataset
# (only needed once; the trained model is saved to data/urgency_model.joblib)
python -m src.train_classifier

# Optional: enable real LLM message generation
export ANTHROPIC_API_KEY=your_key_here   # otherwise falls back to templates

# CLI demo (batch run over all needs.csv rows)
python -m src.main

# Interactive prototype
streamlit run app/app.py
```

## Repository Structure

```
disaster_relief_agent/
├── data/
│   ├── needs.csv
│   ├── resources.csv
│   ├── disaster_tweets.csv     # Kaggle dataset (11,370 labeled tweets)
│   ├── urgency_model.joblib    # trained classifier (created by train_classifier.py)
│   └── agent_logs.db           (created on first run)
├── src/
│   ├── preprocessing.py     # data loading, cleaning, classifier-based text boost
│   ├── train_classifier.py  # trains the urgency/disaster-relevance classifier
│   ├── urgency.py           # urgency scoring & ranking
│   ├── matching.py          # rule-based matching engine
│   ├── guardrails.py        # validation, escalation, human-in-loop rules
│   ├── ai_message.py        # LLM coordination-message generation
│   ├── memory.py            # SQLite logging
│   └── main.py              # agent controller / CLI entry point
├── app/
│   └── app.py                # Streamlit prototype (3 tabs: submit, batch, logs)
├── requirements.txt
└── README.md
```

## Results / Sample Output

Running `python -m src.main` on the sample data matches 14/15 needs
(93% coverage), correctly flags one Chennai medicine request as
`RESOURCE_UNAVAILABLE` (best candidate marked "busy"), and ranks needs by
computed urgency before processing.

## Limitations

- Sample `needs.csv`/`resources.csv` are synthetic, not live data
- The urgency classifier is a simple TF-IDF + Logistic Regression baseline
  (85% accuracy, weaker recall on the minority "disaster" class) — trained
  on generic tweets, not relief-specific need descriptions, so predictions
  may not always transfer perfectly to this domain
- Matching is location + category exact-match only; no fuzzy/geo-distance
  matching yet
- LLM message generation depends on API availability; falls back to a
  fixed template if the API key is missing or the call fails
- This is a decision-support tool only — it never auto-dispatches resources

## Responsible Use

All matches require explicit human approval before any real-world action.
The agent is designed to reduce coordination effort and surface urgent
unmatched needs — not to replace human judgment in a safety-critical domain.

## Future Improvements

- Train a real urgency classifier on the Kaggle disaster-tweets dataset
- Add embeddings-based semantic matching for near-miss categories/locations
- Add a map view and geo-distance-aware matching
- Persist approvals/rejections back into the log for a feedback loop

## Team

_(add team member names here)_
