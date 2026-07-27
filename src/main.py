"""
main.py
-------
Agent controller: ties together preprocessing -> urgency scoring ->
matching -> guardrails -> AI message generation -> logging.

Run directly for a CLI demo:
    python -m src.main
"""

import os
from dotenv import load_dotenv

# Load ANTHROPIC_API_KEY / GROQ_API_KEY from a .env file in the project root,
# same as app.py -- so `python -m src.main` picks up real keys too, not just
# the Streamlit app.
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from src.preprocessing import load_needs, load_resources
from src.urgency import rank_needs_by_urgency
from src.matching import match_need_with_resources
from src.guardrails import apply_guardrails
from src.ai_message import generate_coordination_message
from src.memory import log_decision


def run_agent(needs_path="data/needs.csv", resources_path="data/resources.csv"):
    """
    Runs the full pipeline over all needs and returns a list of
    structured, guardrail-checked results ready for human review.
    """
    needs_df = load_needs(needs_path)
    resources_df = load_resources(resources_path)

    # Step 1: score & rank needs by urgency (highest priority first)
    ranked_needs = rank_needs_by_urgency(needs_df)
    resources = resources_df.to_dict(orient="records")

    results = []
    for need in ranked_needs:
        # Step 2: match
        candidates = match_need_with_resources(need, resources)
        best = candidates[0] if candidates else None

        match_result = {
            "need": need,
            "best_match": best[0] if best else None,
            "score": best[1] if best else 0.0,
        }

        # Step 3: guardrails
        checked = apply_guardrails(match_result)

        # Step 4: AI message (only for matched/reviewable cases)
        message = ""
        if checked["status"] == "MATCHED":
            message = generate_coordination_message(need, checked["resource"], checked["score"])
        elif checked["status"] == "NO_MATCH":
            message = (
                f"No available resource found for {need['category']} in {need['location']}. "
                f"{'ESCALATE to emergency authority.' if checked.get('escalate_to_authority') else 'Log and monitor.'}"
            )

        checked["ai_message"] = message

        # Step 5: log
        log_decision(checked, ai_message=message)

        results.append(checked)

    return results


def print_summary(results):
    print(f"\n{'=' * 60}\nAGENT RUN SUMMARY ({len(results)} needs processed)\n{'=' * 60}")
    for r in results:
        need = r["need"]
        print(f"\n[{r['status']}] {need['category'].title()} in {need['location']} "
              f"(urgency: {need.get('urgency_level', 'N/A')})")
        if r.get("ai_message"):
            print(f"  -> {r['ai_message']}")
        elif r.get("reason"):
            print(f"  -> Reason: {r['reason']}")


if __name__ == "__main__":
    results = run_agent()
    print_summary(results)
