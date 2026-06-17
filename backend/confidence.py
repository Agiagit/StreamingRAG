"""Confidence value and WAIT/SUGGEST/COMMIT decision.

This module is intentionally tiny and dependency-free so the evaluation
work package can tune it without touching retrieval or generation.
All tunable numbers live in the constants block below.
"""

# ---------------------------------------------------------------------------
# Tunable thresholds -- tuned against the full corpus (2 737 chunks / 96 articles)
# with per-article deduplication active.
#
# With dedup, top1 and top2 come from different articles, so the margin
# reflects a real semantic gap. On the full corpus, clear queries (e.g.
# "when was neil armstrong born") yield top1 ~0.38-0.45 and margins
# ~0.10-0.20 when complete. Vague fragments ("arm", "when") yield
# top1 <0.22 and margins near zero. The thresholds below fire COMMIT
# roughly when 2/3 of a clear query has been typed.
# ---------------------------------------------------------------------------
COMMIT_TOP1_THRESHOLD = 0.30   # lowered from 0.35; large diverse corpus scores lower
COMMIT_MARGIN_THRESHOLD = 0.05 # unchanged; dedup makes inter-article margin meaningful
SUGGEST_TOP1_THRESHOLD = 0.22  # lowered from 0.28 for same reason

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
