"""
guardrails.py
-------------
Safety and sanity checks applied before any output is shown to a human.
This is a sensitive domain (disaster relief) -- the agent must never
auto-dispatch resources, must flag unmatched urgent needs for escalation,
and must clearly mark low-confidence matches.
"""

REQUIRED_NEED_FIELDS = ["location", "category", "quantity", "urgency"]
LOW_CONFIDENCE_THRESHOLD = 0.3
ESCALATION_URGENCY_THRESHOLD = 8


def validate_need(need: dict):
    """Check for missing/invalid fields. Returns (is_valid, error_message)."""
    for field in REQUIRED_NEED_FIELDS:
        if field not in need or need[field] in (None, "", "nan"):
            return False, f"Missing required field: '{field}'"
    try:
        if float(need["quantity"]) <= 0:
            return False, "Quantity must be greater than 0"
    except (ValueError, TypeError):
        return False, "Quantity must be numeric"
    return True, None


def apply_guardrails(match_result: dict) -> dict:
    """
    Wraps a raw match result with guardrail flags. Never returns an
    'auto-approved' dispatch -- everything is routed as a recommendation
    that requires human sign-off.
    """
    need = match_result["need"]
    resource = match_result["best_match"]
    score = match_result["score"]

    is_valid, error = validate_need(need)
    if not is_valid:
        return {
            "status": "REJECTED",
            "reason": error,
            "need": need,
            "requires_human_review": True,
        }

    urgency_score = need.get("urgency_score", need.get("urgency", 0))

    if resource is None:
        escalate = urgency_score >= ESCALATION_URGENCY_THRESHOLD
        return {
            "status": "NO_MATCH",
            "reason": "No available resource found for this need's location/category.",
            "need": need,
            "escalate_to_authority": escalate,
            "requires_human_review": True,
        }

    if resource.get("status") != "available":
        return {
            "status": "RESOURCE_UNAVAILABLE",
            "reason": f"Best candidate resource ({resource.get('contact')}) is not marked available.",
            "need": need,
            "candidate": resource,
            "requires_human_review": True,
        }

    if score < LOW_CONFIDENCE_THRESHOLD:
        return {
            "status": "LOW_CONFIDENCE",
            "reason": f"Match confidence ({score}) below threshold ({LOW_CONFIDENCE_THRESHOLD}).",
            "need": need,
            "candidate": resource,
            "requires_human_review": True,
        }

    return {
        "status": "MATCHED",
        "need": need,
        "resource": resource,
        "score": score,
        "requires_human_review": True,  # ALL dispatches require human approval, by design
    }
