"""Independent evaluator primitives for Clover experiments."""

def compare(before: dict, after: dict) -> dict:
    before_score = float(before.get("score", 0))
    after_score = float(after.get("score", 0))
    return {
        "before": before_score,
        "after": after_score,
        "delta": after_score - before_score,
        "improved": after_score > before_score,
    }

def reject_self_certification(claim: dict) -> dict:
    return {
        "accepted": bool(claim.get("independent_evidence")),
        "reason": "Learning claims require evidence outside the claim itself."
    }
