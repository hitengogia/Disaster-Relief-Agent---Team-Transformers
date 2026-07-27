"""
matching.py
-----------
Rule-based matching engine: matches a single need against the resource
pool by location + category, ranks candidates by a confidence score, and
factors in urgency as a tie-breaker.
"""


def confidence_score(need: dict, resource: dict) -> float:
    """
    Score in [0, 1]. Combines:
      - coverage: how much of the requested quantity the resource can fill
      - availability: penalize if resource is not 'available'
    """
    if resource["quantity"] <= 0 or need["quantity"] <= 0:
        return 0.0

    coverage = min(resource["quantity"] / need["quantity"], 1.0)
    availability_penalty = 1.0 if resource["status"] == "available" else 0.3
    return round(coverage * availability_penalty, 3)


def match_need_with_resources(need: dict, resources: list) -> list:
    """
    Find and rank all candidate resources for a given need.
    Returns a list of (resource, score) tuples sorted best-first.
    """
    candidates = []
    for r in resources:
        same_location = r["location"].strip().lower() == need["location"].strip().lower()
        same_category = r["category"].strip().lower() == need["category"].strip().lower()
        if same_location and same_category:
            score = confidence_score(need, r)
            if score > 0:
                candidates.append((r, score))

    return sorted(candidates, key=lambda x: x[1], reverse=True)


def match_all_needs(needs: list, resources: list) -> list:
    """
    Match every need against the resource pool.
    Returns a list of dicts: {need, best_match, score, all_candidates}
    """
    results = []
    for need in needs:
        candidates = match_need_with_resources(need, resources)
        best = candidates[0] if candidates else None
        results.append({
            "need": need,
            "best_match": best[0] if best else None,
            "score": best[1] if best else 0.0,
            "all_candidates": candidates,
        })
    return results
