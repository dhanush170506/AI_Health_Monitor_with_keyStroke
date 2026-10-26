"""
AI Health Monitor - Health Predictor + Score Fusion (Modules 6 & 8)
====================================================================

Loads model/health_model.pkl, runs predict_proba on each minute's feature
vector, and fuses:

  1. ML class probabilities          (primary supervised signal)
  2. Personal baseline deviations    (personalization adjustment)
  3. Motor anomaly score             (Module 7, from baseline deviations)

into final 0-100 scores: fatigue, stress, anxiety, motor_risk, plus overall
status and non-medical recommendations.

HONESTY: the model is trained on synthetic behavioural data. Nothing here
diagnoses any medical condition.
"""

import json
import math
import os
import threading

MODEL_PATH = os.path.join("model", "health_model.pkl")
FEATURE_NAMES_PATH = os.path.join("model", "feature_names.json")

# Fallback feature names used only if the saved metadata file is missing.
DEFAULT_FEATURE_NAMES = [
    "typing_speed_wpm", "avg_dwell_time", "dwell_std", "avg_flight_time",
    "flight_std", "backspace_rate", "error_rate", "typing_pause_duration",
    "mouse_speed", "mouse_speed_std", "mouse_acceleration",
    "mouse_pause_duration", "mouse_jitter", "mouse_efficiency", "click_rate",
    "target_acquisition_time",
]

CLASS_LABELS = ["Normal", "Fatigue", "Stress", "Anxiety"]  # 0,1,2,3

# Mapping model-class -> fused score direction (which baseline deviations
# support which health score).
DEVIATION_SUPPORT = {
    "Fatigue": [("typing_speed_wpm", -1.0), ("avg_dwell_time", 1.0),
                ("avg_flight_time", 1.0), ("typing_pause_duration", 1.0),
                ("mouse_speed", -1.0)],
    "Stress": [("typing_speed_wpm", 1.0), ("backspace_rate", 1.0),
               ("error_rate", 1.0), ("mouse_speed", 1.0),
               ("mouse_speed_std", 1.0)],
    "Anxiety": [("dwell_std", 1.0), ("flight_std", 1.0),
                ("mouse_jitter", 1.0), ("mouse_speed_std", 1.0),
                ("mouse_acceleration", 1.0)],
}

MOTOR_KEYS = [
    ("dwell_std", 1.0),
    ("avg_flight_time", 1.0),
    ("flight_std", 1.0),
    ("mouse_jitter", 1.0),
    ("mouse_efficiency", -1.0),
    ("mouse_speed_std", 1.0),
    ("target_acquisition_time", 1.0),
]

ZSAT = 2.0          # |z| at which a feature's support contribution saturates
W_MODEL = 0.70      # ML probabilities are the primary signal
W_BASE = 0.30       # baseline deviation adjustment
W_MOTOR_IN_OVERALL = 0.50  # motor risk weight inside the overall status


class Predictor:
    """Thread-safe wrapper around the trained RandomForest model."""

    def __init__(self, model_path=MODEL_PATH, feature_names_path=FEATURE_NAMES_PATH):
        self.model = None
        self.feature_names = list(DEFAULT_FEATURE_NAMES)
        self.model_path = model_path
        self.feature_names_path = feature_names_path
        self._lock = threading.Lock()
        self.load()

    def load(self):
        try:
            import joblib
            with self._lock:
                self.model = joblib.load(self.model_path)
                # trust saved feature names when available
                try:
                    with open(self.feature_names_path, "r", encoding="utf-8") as f:
                        self.feature_names = json.load(f)
                except (OSError, ValueError):
                    pass
            return True
        except (OSError, ValueError, ImportError):
            self.model = None
            return False

    @property
    def ready(self):
        return self.model is not None

    def predict_proba(self, vector):
        """Return {class_name: probability} using canonical feature order."""
        if self.model is None:
            return {c: 0.25 for c in CLASS_LABELS}
        row = [[float(vector.get(k, 0.0)) for k in self.feature_names]]
        with self._lock:
            proba = self.model.predict_proba(row)[0]
        classes = [str(c) for c in self.model.classes_]
        out = {c: 0.0 for c in CLASS_LABELS}
        for cls, p in zip(classes, proba):
            name = CLASS_LABELS[int(cls)] if cls.isdigit() else cls
            out[name] = float(p)
        return out

    # -- fusion ------------------------------------------------------------

    def analyze(self, vector, deviations, motor_risk):
        """Fuse ML probabilities + baseline deviations + motor risk.

        Returns a dict with the four health scores (0-100), overall status,
        explanation lines and a non-medical recommendation.
        """
        proba = self.predict_proba(vector)

        def dev_support(class_name):
            s = 0.0
            total_w = 0.0
            for key, direction in DEVIATION_SUPPORT.get(class_name, []):
                z = deviations.get(key, 0.0)
                contribution = max(-1.0, min(1.0, z / ZSAT)) * direction
                s += contribution
                total_w += 1.0
            return s / total_w if total_w else 0.0

        def clamp01(x):
            return max(0.0, min(1.0, x))

        # Multiplicative fusion: baseline deviation acts as a confidence
        # amplifier/dampener on the model probability, rather than adding a
        # fixed offset. A weak model signal cannot be inflated into a high
        # score by deviation alone (and vice versa).
        fused = {}
        for cls, p in proba.items():
            support = dev_support(cls)             # -1..+1
            gain = 1.0 + W_BASE * 0.5 * support    # 0.85 .. 1.15
            combined = clamp01(p * gain)
            fused[cls] = int(round(combined * 100))

        # Motor risk comes straight from Module 7 and joins the overall status.
        scores = {
            "fatigue_score": fused.get("Fatigue", 0),
            "stress_score": fused.get("Stress", 0),
            "anxiety_score": fused.get("Anxiety", 0),
            "motor_risk_score": int(round(max(0, min(100, motor_risk)))),
        }

        overall = self._overall_status(scores, dev_support)
        explanation, indicators = self._explain(vector, deviations, motor_risk, scores)
        recommendation = self._recommend(overall, scores, deviations)

        return {
            **scores,
            "overall_status": overall,
            "explanation": explanation,
            "indicators": indicators,
            "recommendation": recommendation,
            "ml_probabilities": {k: round(v, 4) for k, v in proba.items()},
        }

    # -- helpers -------------------------------------------------------------

    def _overall_status(self, scores, dev_support):
        def band(v):
            return "NORMAL" if v < 30 else ("MODERATE" if v < 60 else "ELEVATED")

        health = [scores["fatigue_score"], scores["stress_score"], scores["anxiety_score"]]
        # Motor risk is a distinct indicator - give it its own weight in overall.
        base = max(health) * 0.6 + (sum(health) / len(health)) * 0.4
        motor = scores["motor_risk_score"]
        overall = base * (1 - W_MOTOR_IN_OVERALL) + motor * W_MOTOR_IN_OVERALL
        return band(overall)

    def _explain(self, vector, deviations, motor_risk, scores):
        lines = []
        indicators = []

        def line(cond, text):
            if cond:
                lines.append(text)
                return True
            return False

        def ind(cond, text):
            if cond:
                indicators.append(text)

        wpm_z = deviations.get("typing_speed_wpm", 0.0)
        dwell_z = deviations.get("avg_dwell_time", 0.0)
        dwell_std_z = deviations.get("dwell_std", 0.0)
        flight_z = deviations.get("avg_flight_time", 0.0)
        jitter_z = deviations.get("mouse_jitter", 0.0)
        eff_z = deviations.get("mouse_efficiency", 0.0)
        speed_std_z = deviations.get("mouse_speed_std", 0.0)
        bs_z = deviations.get("backspace_rate", 0.0)
        mspeed_z = deviations.get("mouse_speed", 0.0)
        err_z = deviations.get("error_rate", 0.0)
        tpause_z = deviations.get("typing_pause_duration", 0.0)
        acq_z = deviations.get("target_acquisition_time", 0.0)

        line(abs(wpm_z) < 0.75 and abs(dwell_z) < 0.75,
             "Your recent behaviour is close to your personal baseline.")
        line(wpm_z <= -1.0, "Typing speed has decreased compared with your baseline.")
        line(wpm_z >= 1.0, "Typing speed is above your personal baseline.")
        line(dwell_z >= 1.0, "Key press (dwell) times are longer than usual.")
        line(bs_z >= 1.0, "More corrections (backspaces) than your normal pattern.")
        line(err_z >= 1.0, "Typing error rate is above your normal pattern.")
        line(tpause_z >= 1.0, "Longer mid-typing pauses than usual.")
        line(mspeed_z >= 1.0, "Mouse movements are faster than your normal pattern.")
        line(mspeed_z <= -1.0, "Mouse movements are slower than your normal pattern.")
        line(acq_z >= 1.0, "Slower target acquisition (reaction/motor) than usual.")

        # Module 7 style indicator strings, shown only when actually elevated.
        ind(dwell_std_z >= 1.0, "Increased dwell-time variability")
        ind(flight_z >= 1.0 or deviations.get("flight_std", 0.0) >= 1.0,
            "Flight-time slowing / increased key-gap variability")
        ind(jitter_z >= 1.0, "Increased mouse movement jitter")
        ind(eff_z <= -1.0, "Reduced movement efficiency")
        ind(speed_std_z >= 1.0, "Increased mouse speed variation")
        ind(err_z >= 1.0, "Increased typing error rate")
        ind(tpause_z >= 1.0, "Increased typing hesitation/pauses")

        if not indicators and motor_risk >= 30:
            indicators.append("Minor deviations across motor-related features")

        if not lines:
            lines.append("Behaviour shows mixed deviations from your baseline.")
        return lines, indicators

    def _recommend(self, overall, scores, deviations):
        if overall == "NORMAL":
            return ("Your recent behaviour is close to your personal baseline. "
                    "Keep working comfortably.")
        if overall == "MODERATE":
            if scores["fatigue_score"] >= scores["stress_score"] and \
               scores["fatigue_score"] >= scores["anxiety_score"]:
                return ("Consider taking a short break if you have been working "
                        "continuously; slight fatigue signals detected.")
            if scores["anxiety_score"] >= max(scores["stress_score"], scores["fatigue_score"]):
                return ("A short pause or a few slow breaths may help; slight "
                        "restlessness signals detected.")
            return ("Consider a brief pause and a stretch; your activity pattern "
                    "is somewhat elevated versus baseline.")
        # ELEVATED
        return ("Take a proper break, hydrate, and resume at a comfortable pace. "
                "This prototype flags behavioural deviation only - it is not a "
                "medical assessment.")
