"""
AI Health Monitor - Analyzer (Modules 7 & 8)
============================================

* Motor-symptom RISK indicator (NOT a Parkinson's or any diagnosis): a
  lightweight statistical anomaly score from personal-baseline deviations of
  motor-related features (dwell variability, flight-time slowing, mouse jitter,
  movement efficiency, mouse-speed variation).
* The single analyze() entry point wiring feature vector -> predictor ->
  baseline deviation fusion -> final dashboard payload.

Class bands for every score:  0-29 LOW / NORMAL, 30-59 MODERATE, 60-100 ELEVATED.
"""

def band_label(value):
    v = max(0, min(100, int(value)))
    if v < 30:
        return "LOW"
    if v < 60:
        return "MODERATE"
    return "ELEVATED"

OVERALL_BANDS = {"NORMAL": "NORMAL", "MODERATE": "MODERATE", "ELEVATED": "ELEVATED"}


def motor_anomaly_score(deviations):
    """Compute 0-100 motor anomaly risk from signed baseline deviations.

    Elevating features (positive z raises risk): dwell variability, flight
    slowing / variability, mouse jitter, mouse speed variation.
    Lowering features (negative z raises risk): movement efficiency.

    |z| = 2.0 saturates one feature's contribution. Weighted mean of
    contributions -> 0..1 -> 0..100. With no baseline the score is neutral-low.
    """
    contributions = [
        ("dwell_std", 1.0, +1.0),
        ("avg_flight_time", 0.8, +1.0),
        ("flight_std", 1.0, +1.0),
        ("mouse_jitter", 1.2, +1.0),
        ("mouse_efficiency", 1.0, -1.0),
        ("mouse_speed_std", 0.8, +1.0),
    ]
    sat = 2.0
    total_w = 0.0
    acc = 0.0
    for key, w, direction in contributions:
        z = deviations.get(key, 0.0)
        c = max(-1.0, min(1.0, z / sat)) * direction  # -1..1
        acc += max(0.0, c) * w                        # only elevation counts
        total_w += w
    raw = acc / total_w if total_w else 0.0
    # Gentle mapping: broad 1-sigma elevation lands in the low-Moderate band,
    # consistent 2-sigma elevation approaches the top of the scale.
    score = 100.0 * raw ** 1.3
    return int(round(max(0, min(100, score))))


def analyze(vector, deviations, predictor, baseline_established):
    """Run the full analysis for one 1-minute window.

    Returns the dashboard-ready result dict. Always succeeds: with no model or
    no baseline it degrades gracefully.
    """
    motor = motor_anomaly_score(deviations)

    if not baseline_established:
        # Without a personal baseline the fusion is model-only.
        proba = predictor.predict_proba(vector)
        scores = {
            "fatigue_score": int(round(proba.get("Fatigue", 0.0) * 100)),
            "stress_score": int(round(proba.get("Stress", 0.0) * 100)),
            "anxiety_score": int(round(proba.get("Anxiety", 0.0) * 100)),
            "motor_risk_score": motor,
        }
        result = {
            **scores,
            "overall_status": band_label(max(scores.values())),
            "explanation": ["Personal baseline not yet established - "
                            "scores are model-only estimates."],
            "indicators": [],
            "recommendation": ("Continue typing and moving normally; the personal "
                               "baseline will sharpen these estimates."),
            "ml_probabilities": {k: round(v, 4) for k, v in proba.items()},
        }
    else:
        result = predictor.analyze(vector, deviations, motor)

    # Attach convenience bands + labels for the dashboard cards.
    for key, prefix in (("fatigue_score", "fatigue"), ("stress_score", "stress"),
                        ("anxiety_score", "anxiety"),
                        ("motor_risk_score", "motor")):
        result[f"{prefix}_band"] = band_label(result[key])
    return result
