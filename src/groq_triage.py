"""
groq_triage.py
--------------
Fast first-pass triage using Groq (open-weight models served at very high
inference speed). This is the second half of the dual-model architecture:

    Groq  -> fast, cheap triage: sanity-check category, quick urgency read
    Claude -> slower, higher-quality final coordination message (ai_message.py)

Rationale (matters for the "clear purpose" requirement): calling two LLMs
only makes sense if they're doing different jobs. Groq's job here is
low-latency structured triage that can run on *every* submission, even
ones that never reach the matching/messaging stage (e.g. to give instant
UI feedback while the rest of the pipeline runs). Claude's job is the
final natural-language message a human coordinator reads. Neither call is
decorative — the triage output is optionally fed into the urgency boost
and into the final Claude prompt as an extra signal.

As with ai_message.py, this must never break the app: no GROQ_API_KEY, an
API error, or a malformed response all fall back to a deterministic
keyword-based triage so the pipeline (and any live demo) keeps working.
"""

import os
import json

FAST_MODEL = "openai/gpt-oss-20b"  # llama-3.1-8b-instant was deprecated by Groq
                                     # (June 17 2026); this is Groq's recommended
                                     # replacement for the same use case.

TRIAGE_SYSTEM_PROMPT = (
    "You are a fast triage classifier for a disaster-relief intake system. "
    "Given a short need description, respond with ONLY a JSON object, no "
    "prose, no markdown fences, in this exact shape:\n"
    '{"urgency_hint": "LOW"|"MEDIUM"|"HIGH", "category_guess": string, '
    '"reasoning": string (max 15 words)}\n'
    "category_guess must be one of: food, medicine, shelter, water, transport, other."
)

HIGH_URGENCY_KEYWORDS = [
    "trapped", "dying", "collapsed", "flooding", "stranded", "critical",
    "injured", "rescue", "sos", "life threatening", "no food", "no water",
]


def _keyword_fallback(description: str, category: str = "") -> dict:
    """Deterministic triage used when Groq is unavailable — mirrors the
    same style of fallback used elsewhere in this project (preprocessing.py,
    ai_message.py) so the app degrades predictably, not silently."""
    text = (description or "").lower()
    hits = sum(1 for kw in HIGH_URGENCY_KEYWORDS if kw in text)
    urgency_hint = "HIGH" if hits >= 2 else "MEDIUM" if hits == 1 else "LOW"
    return {
        "urgency_hint": urgency_hint,
        "category_guess": category or "other",
        "reasoning": "keyword heuristic (Groq unavailable)",
        "source": "fallback",
    }


def fast_triage(need: dict) -> dict:
    """
    Run a fast triage pass on a need's free-text description via Groq.

    Returns:
        {"urgency_hint": "LOW"|"MEDIUM"|"HIGH", "category_guess": str,
         "reasoning": str, "source": "groq"|"fallback"}

    Never raises — falls back to a keyword heuristic on any failure
    (missing key, network error, malformed response), matching the
    fail-soft pattern used by generate_coordination_message().
    """
    description = need.get("description", "")
    category = need.get("category", "")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or not description or not description.strip():
        return _keyword_fallback(description, category)

    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=FAST_MODEL,
            max_tokens=100,
            temperature=0,
            messages=[
                {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
                {"role": "user", "content": description},
            ],
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)

        return {
            "urgency_hint": parsed.get("urgency_hint", "MEDIUM"),
            "category_guess": parsed.get("category_guess", category or "other"),
            "reasoning": parsed.get("reasoning", ""),
            "source": "groq",
        }
    except Exception as e:
        # Covers: missing groq package, network error, rate limit,
        # non-JSON response, unexpected schema, deprecated/invalid model ID.
        # Never break the pipeline -- but DO print the real reason to the
        # terminal, since silently swallowing it makes real key/model
        # problems indistinguishable from "no key set" in the UI.
        print(f"[groq_triage] Groq call failed, using keyword fallback: {e}")
        fallback = _keyword_fallback(description, category)
        fallback["reasoning"] = "keyword heuristic (Groq call failed)"
        return fallback


if __name__ == "__main__":
    sample_need = {
        "category": "water",
        "description": "Families trapped on rooftops, no clean water for two days",
    }
    result = fast_triage(sample_need)
    print(json.dumps(result, indent=2))
