"""Confidence value and WAIT/SUGGEST/COMMIT decision.

This module is intentionally tiny and dependency-free so the evaluation
work package can tune it without touching retrieval or generation.
All tunable numbers live in the constants block below.
"""

# ---------------------------------------------------------------------------
# Tunable thresholds (placeholders, to be adjusted by the evaluation WP).
# ---------------------------------------------------------------------------
COMMIT_TOP1_THRESHOLD = 0.4   # top1 similarity needed to commit
COMMIT_MARGIN_THRESHOLD = 0.04  # top1 - top2 margin needed to commit
SUGGEST_TOP1_THRESHOLD = 0.3  # top1 similarity needed to suggest

# Weights for blending the two signals into a single confidence value.
# top1 carries most of the weight; the margin tops it up when the best
# hit clearly separates from the runner-up.
CONFIDENCE_TOP1_WEIGHT = 0.7
CONFIDENCE_MARGIN_WEIGHT = 0.3


def score_query(top1_score: float, top2_score: float) -> tuple[float, str]:
    """Return (confidence, decision) for the two best similarity scores.

    decision is exactly one of "WAIT", "SUGGEST", "COMMIT".
    """
    margin = top1_score - top2_score

    # Margin saturates at the commit threshold: any separation beyond what
    # we require for a commit should not keep inflating confidence.
    margin_signal = min(max(margin / COMMIT_MARGIN_THRESHOLD, 0.0), 1.0)
    top1_signal = min(max(top1_score, 0.0), 1.0)

    confidence = (
        CONFIDENCE_TOP1_WEIGHT * top1_signal
        + CONFIDENCE_MARGIN_WEIGHT * margin_signal
    )
    confidence = round(min(max(confidence, 0.0), 1.0), 4)

    if top1_score >= COMMIT_TOP1_THRESHOLD and margin >= COMMIT_MARGIN_THRESHOLD:
        decision = "COMMIT"
    elif top1_score >= SUGGEST_TOP1_THRESHOLD:
        decision = "SUGGEST"
    else:
        decision = "WAIT"

    return confidence, decision
