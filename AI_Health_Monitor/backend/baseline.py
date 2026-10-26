"""
AI Health Monitor - Personal Behavioural Baseline (Module 5)
============================================================

Learns the user's "normal" behaviour during a calibration period, persists it
to data/baseline.json, and computes standardized deviations of new feature
vectors against that baseline.

Deviation convention: POSITIVE z = above personal norm, NEGATIVE z = below.

Safe handling built in:
  * zero / tiny standard deviations get a floor before dividing
  * incremental updates use a shadow statistic with outlier damping so one
    abnormal session cannot overwrite a healthy baseline
"""

import json
import math
import os
import threading

FEATURE_KEYS = [
    "typing_speed_wpm",
    "avg_dwell_time",
    "dwell_std",
    "avg_flight_time",
    "flight_std",
    "backspace_rate",
    "error_rate",
    "typing_pause_duration",
    "mouse_speed",
    "mouse_speed_std",
    "mouse_acceleration",
    "mouse_pause_duration",
    "mouse_jitter",
    "mouse_efficiency",
    "click_rate",
    "target_acquisition_time",
]

# v2: 16-feature schema (added error_rate, typing_pause_duration,
# mouse_acceleration, target_acquisition_time). v1 files are ignored.
BASELINE_SCHEMA_VERSION = 2

MIN_SAMPLES_FOR_BASELINE = 2   # minimum calibration windows before established
STD_FLOOR = 1e-6
REL_STD_FLOOR = 0.15           # std >= 15% of |mean| (early-calibration guard)
Z_FLOOR = 1e-3                 # avoid exact-zero deviations downstream


class Baseline:
    """Holds calibration samples and the derived mean/std statistics."""

    def __init__(self, path, calibration_seconds=120.0):
        self.path = path
        self.calibration_seconds = float(calibration_seconds)
        self._lock = threading.Lock()
        self.samples = []          # list of 12-float feature vectors
        self.mean = {k: 0.0 for k in FEATURE_KEYS}
        self.std = {k: 1.0 for k in FEATURE_KEYS}
        self.established = False

    # -- persistence --------------------------------------------------------

    def save(self):
        try:
            with self._lock:
                payload = {
                    "version": BASELINE_SCHEMA_VERSION,
                    "n_samples": len(self.samples),
                    "mean": self.mean,
                    "std": self.std,
                    "established": self.established,
                }
                tmp = self.path + ".tmp"
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                os.replace(tmp, self.path)
        except OSError:
            pass  # a failed persistence must never crash monitoring

    def load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if payload.get("version") == BASELINE_SCHEMA_VERSION:
                self.mean = {k: float(payload["mean"].get(k, 0.0)) for k in FEATURE_KEYS}
                self.std = {k: max(float(payload["std"].get(k, 1.0)), STD_FLOOR)
                            for k in FEATURE_KEYS}
                self.established = bool(payload.get("established"))
                return True
        except (OSError, ValueError, TypeError):
            pass
        return False

    def reset(self):
        with self._lock:
            self.samples = []
            self.established = False
        if os.path.exists(self.path):
            try:
                os.remove(self.path)
            except OSError:
                pass

    # -- calibration --------------------------------------------------------

    def add_calibration_sample(self, vector):
        """Add one minute-window feature vector during calibration."""
        with self._lock:
            self.samples.append([float(vector[k]) for k in FEATURE_KEYS])
            n = len(self.samples)
            if n >= MIN_SAMPLES_FOR_BASELINE:
                self._recompute_locked()
                self.established = True
        self.save()

    def _recompute_locked(self):
        n = len(self.samples)
        for i, k in enumerate(FEATURE_KEYS):
            vals = [s[i] for s in self.samples]
            m = sum(vals) / n
            var = sum((v - m) ** 2 for v in vals) / max(n - 1, 1)
            self.mean[k] = m
            # Floor tiny std-devs: zero-variance features would explode z-scores.
            # A relative floor (15% of the mean) keeps early, noisy calibrations
            # from producing absurd deviation scores.
            self.std[k] = max(math.sqrt(var), REL_STD_FLOOR * abs(m), STD_FLOOR)

    # -- deviation scoring ---------------------------------------------------

    def deviations(self, vector):
        """Return dict feature -> signed z-score vs. personal baseline."""
        out = {}
        for k in FEATURE_KEYS:
            s = max(self.std.get(k, STD_FLOOR), STD_FLOOR)
            z = (float(vector.get(k, 0.0)) - self.mean.get(k, 0.0)) / s
            # tiny-deviation floor: avoid meaningless 0.0 z for near-identical
            # values when std is microscopic, but keep signed direction
            out[k] = math.copysign(max(abs(z), 0.0), z) if abs(z) > Z_FLOOR else 0.0
        return out

    # -- incremental update ---------------------------------------------------

    def update_incremental(self, vector, z_max=3.0, rate=0.05):
        """Gently blend a new *normal-ish* window into the baseline.

        Windows whose worst |z| exceeds z_max are considered anomalous and are
        ignored, so an abnormal session cannot drag the baseline towards it.
        """
        if not self.established:
            return
        devs = self.deviations(vector)
        worst = max(abs(z) for z in devs.values()) if devs else 0.0
        if worst > z_max:
            return  # abnormal window: do not contaminate the baseline
        with self._lock:
            for k in FEATURE_KEYS:
                self.mean[k] = (1 - rate) * self.mean[k] + rate * float(vector[k])
                # slowly relax std towards a sane floor, never towards 0
                self.std[k] = max(self.std[k] * (1 - rate * 0.5) +
                                  abs(devs[k]) * self.std[k] * rate * 0.5,
                                  REL_STD_FLOOR * abs(self.mean[k]), STD_FLOOR)
        self.save()

    # -- display helpers -------------------------------------------------------

    def summary(self):
        return {
            "established": self.established,
            "calibration_seconds": self.calibration_seconds,
            "n_samples": len(self.samples),
            "mean": {k: round(self.mean[k], 3) for k in FEATURE_KEYS},
            "std": {k: round(self.std[k], 3) for k in FEATURE_KEYS},
            "display": {
                "typing_speed_wpm": round(self.mean["typing_speed_wpm"], 1),
                "avg_dwell_time": round(self.mean["avg_dwell_time"], 1),
                "avg_flight_time": round(self.mean["avg_flight_time"], 1),
                "error_rate": round(self.mean["error_rate"], 2),
                "mouse_speed": round(self.mean["mouse_speed"], 1),
                "mouse_jitter": round(self.mean["mouse_jitter"], 4),
                "target_acquisition_time": round(self.mean["target_acquisition_time"], 0),
            },
        }
