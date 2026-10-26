"""Mode-aware behavioural deviations and cross-device indicator scoring."""

import json
import logging
import math
import os

import numpy as np
from sklearn.cluster import KMeans

LOGGER = logging.getLogger(__name__)
MIN_MODE_WINDOWS = 15
MAX_MODE_OBSERVATIONS = 180
MIN_BASELINE_WINDOWS_PER_MODE = 5
Z_EPSILON = 1e-6
Z_LIMIT = 3.0
AGREEMENT_THRESHOLD = 0.5
ALPHA = 0.02
BETA = 0.3

# Each feature weight is normalized within its device contribution. Overall
# deviation uses the same raw weights divided once by their available total.
INDICATORS = {
    "fatigue": {
        "keyboard": [
            ("typing_speed_wpm", 0.25, -1.0),
            ("typing_pause_duration", 0.25, 1.0),
            ("avg_dwell_time", 0.15, 1.0),
            ("flight_std", 0.15, 1.0),
        ],
        "mouse": [("mouse_average_velocity", 0.20, -1.0)],
    },
    "stress_tension": {
        "keyboard": [
            ("backspace_rate", 0.25, 1.0),
            ("flight_std", 0.25, 1.0),
        ],
        "mouse": [
            ("mouse_jitter_index", 0.20, 1.0),
            ("mouse_direction_changes", 0.15, 1.0),
            ("mouse_click_interval_variability", 0.15, 1.0),
        ],
    },
    "motor_consistency": {
        "keyboard": [("dwell_std", 0.20, 1.0)],
        "mouse": [
            ("mouse_jitter_index", 0.30, 1.0),
            ("mouse_path_efficiency", 0.25, -1.0),
            ("mouse_acceleration_variability", 0.25, 1.0),
        ],
    },
}


def interaction_vector(features, meta, window_seconds=60.0):
    """Create the required event-only mode signature."""
    minutes = max(float(window_seconds) / 60.0, Z_EPSILON)
    key_rate = max(0.0, float(meta.get("n_key_events", 0))) / minutes
    path_rate = max(0.0, float(features.get("mouse_path_length") or 0.0)) / minutes
    click_rate = max(0.0, float(meta.get("n_clicks", 0))) / minutes
    scrolls = max(0.0, float(meta.get("n_scroll_events", 0)))
    mouse_events = max(0.0, float(meta.get("n_mouse_points", 0)))
    scroll_share = scrolls / max(mouse_events + scrolls, 1.0)
    return np.asarray(
        [math.log1p(key_rate), math.log1p(path_rate), click_rate, scroll_share],
        dtype=float,
    )


class BehaviouralAnalysis:
    """Learn per-mode baselines and compute guarded behavioural indicators.

    Persisted data contains aggregate feature statistics only; raw trajectories
    are never accepted or stored by this class.
    """

    def __init__(self, path):
        self.path = path
        self.observations = []
        self.model = None
        self.centroids = None
        self.acceptance_distance = None
        self.cluster_labels = {}
        self.mode_stats = {}
        self.mode_window_counts = {}
        self.fast_averages = {}
        self.rising_reports = {}
        self.baseline_frozen = False
        self.load()

    def load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as stream:
                payload = json.load(stream)
        except FileNotFoundError:
            return False
        except (OSError, ValueError) as exc:
            LOGGER.warning("Unable to load behavioural baseline: %s", exc)
            return False
        if not isinstance(payload, dict) or payload.get("version") != 1:
            return False
        self.observations = payload.get("observations", [])
        self.centroids = payload.get("centroids")
        self.acceptance_distance = payload.get("acceptance_distance")
        self.cluster_labels = payload.get("cluster_labels", {})
        self.mode_stats = payload.get("mode_stats", {})
        self.mode_window_counts = payload.get("mode_window_counts", {})
        self.fast_averages = payload.get("fast_averages", {})
        self.rising_reports = payload.get("rising_reports", {})
        self.baseline_frozen = bool(payload.get("baseline_frozen", False))
        if self.centroids:
            self.centroids = np.asarray(self.centroids, dtype=float)
        return True

    def reset(self):
        self.observations = []
        self.model = None
        self.centroids = None
        self.acceptance_distance = None
        self.cluster_labels = {}
        self.mode_stats = {}
        self.mode_window_counts = {}
        self.fast_averages = {}
        self.rising_reports = {}
        self.baseline_frozen = False
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            LOGGER.warning("Unable to remove behavioural baseline: %s", exc)

    def summary(self):
        counts = [
            {
                "mode": self.cluster_labels.get(cluster, "unknown"),
                "windows": count,
            }
            for cluster, count in self.mode_window_counts.items()
        ]
        ready = (
            self.centroids is not None
            and len(self.mode_window_counts) == 3
            and all(count >= MIN_BASELINE_WINDOWS_PER_MODE
                    for count in self.mode_window_counts.values())
        )
        return {
            "established": ready,
            "minimum_windows_per_mode": MIN_BASELINE_WINDOWS_PER_MODE,
            "windows_by_mode": counts,
        }

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        payload = {
            "version": 1,
            "observations": self.observations,
            "centroids": self.centroids.tolist() if self.centroids is not None else None,
            "acceptance_distance": self.acceptance_distance,
            "cluster_labels": self.cluster_labels,
            "mode_stats": self.mode_stats,
            "mode_window_counts": self.mode_window_counts,
            "fast_averages": self.fast_averages,
            "rising_reports": self.rising_reports,
            "baseline_frozen": self.baseline_frozen,
        }
        temporary = self.path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
        os.replace(temporary, self.path)

    def _fit_modes(self):
        signatures = np.asarray([item["signature"] for item in self.observations],
                                dtype=float)
        if np.unique(signatures, axis=0).shape[0] < 3:
            return False
        model = KMeans(n_clusters=3, random_state=42, n_init=10)
        cluster_ids = model.fit_predict(signatures)
        distances = model.transform(signatures).min(axis=1)
        self.model = model
        self.centroids = model.cluster_centers_
        # A small floor keeps zero-spread calibration from rejecting tiny jitter.
        self.acceptance_distance = max(
            float(np.mean(distances) + 3 * np.std(distances)),
            float(np.max(distances) * 1.25),
            0.05,
        )

        key_mean = float(np.mean(self.centroids[:, 0]))
        path_mean = float(np.mean(self.centroids[:, 1]))
        self.cluster_labels = {}
        for cluster_id, centroid in enumerate(self.centroids):
            delta = centroid[0] - centroid[1]
            if delta > 0.25 and centroid[0] >= key_mean:
                label = "typing-dominant"
            elif delta < -0.25 and centroid[1] >= path_mean:
                label = "pointer-dominant"
            else:
                label = "mixed"
            self.cluster_labels[str(cluster_id)] = label
            grouped = [item["features"] for item, assigned in
                       zip(self.observations, cluster_ids) if assigned == cluster_id]
            self.mode_stats[str(cluster_id)] = self._feature_stats(grouped)
            self.mode_window_counts[str(cluster_id)] = len(grouped)
        self.observations.clear()
        return True

    def _add_mode_calibration(self, cluster_id, features):
        stats = self.mode_stats[str(cluster_id)]
        for key, raw in features.items():
            if raw is None or not isinstance(raw, (int, float)) or not math.isfinite(raw):
                continue
            value = float(raw)
            stat = stats.get(key)
            if stat is None:
                stats[key] = {"count": 1, "mean": value, "variance": 0.0}
                continue
            old_count = stat["count"]
            delta = value - stat["mean"]
            stat["count"] += 1
            stat["mean"] += delta / stat["count"]
            stat["variance"] = max(
                ((old_count - 1) * stat["variance"] + delta *
                 (value - stat["mean"])) / max(stat["count"] - 1, 1),
                0.0,
            )
        self.mode_window_counts[str(cluster_id)] += 1

    @staticmethod
    def _feature_stats(feature_rows):
        keys = sorted({key for row in feature_rows for key, value in row.items()
                       if value is not None and isinstance(value, (int, float))
                       and math.isfinite(value)})
        stats = {}
        for key in keys:
            values = [float(row[key]) for row in feature_rows
                      if row.get(key) is not None and math.isfinite(float(row[key]))]
            mean = float(np.mean(values))
            variance = float(np.var(values, ddof=1)) if len(values) > 1 else 0.0
            stats[key] = {"count": len(values), "mean": mean,
                          "variance": max(variance, 0.0)}
        return stats

    def _mode_for(self, signature):
        distances = np.linalg.norm(self.centroids - signature, axis=1)
        cluster_id = int(np.argmin(distances))
        distance = float(distances[cluster_id])
        label = self.cluster_labels[str(cluster_id)]
        return cluster_id, label, distance, distance <= self.acceptance_distance

    def _z_scores(self, features, cluster_id):
        stats = self.mode_stats.get(str(cluster_id), {})
        values = {}
        for key, stat in stats.items():
            raw = features.get(key)
            if raw is None or stat["count"] < MIN_BASELINE_WINDOWS_PER_MODE:
                continue
            std = math.sqrt(max(float(stat["variance"]), 0.0))
            value = (float(raw) - float(stat["mean"])) / max(std, Z_EPSILON)
            values[key] = max(-Z_LIMIT, min(Z_LIMIT, value))
        return values

    @staticmethod
    def _contribution(feature_weights, z_scores):
        available = [(weight, sign * z_scores[key])
                     for key, weight, sign in feature_weights
                     if key in z_scores]
        if not available:
            return None
        return sum(weight * value for weight, value in available) / sum(
            weight for weight, _ in available
        )

    def _score_indicators(self, z_scores):
        result = {}
        for name, devices in INDICATORS.items():
            keyboard = self._contribution(devices["keyboard"], z_scores)
            mouse = self._contribution(devices["mouse"], z_scores)
            available = [
                (weight, sign * z_scores[key])
                for features in devices.values()
                for key, weight, sign in features
                if key in z_scores
            ]
            score = (sum(weight * value for weight, value in available)
                     / sum(weight for weight, _ in available)
                     if available else None)
            if keyboard is None or mouse is None:
                status = "Insufficient data"
                corroborated = False
                percent = None
            elif keyboard >= AGREEMENT_THRESHOLD and mouse >= AGREEMENT_THRESHOLD:
                corroborated = True
                percent = round(100 * max(0.0, min(1.0, score / 3.0)))
                status = "Notable change" if score >= 1.5 else "Mild change"
            elif keyboard >= AGREEMENT_THRESHOLD:
                corroborated, percent = False, None
                status = "Keyboard-only change"
            elif mouse >= AGREEMENT_THRESHOLD:
                corroborated, percent = False, None
                status = "Mouse-only change"
            else:
                corroborated = False
                percent = 0
                status = "Normal"
            result[name] = {
                "keyboard_deviation": keyboard,
                "mouse_deviation": mouse,
                "combined_deviation": score,
                "score": percent,
                "status": status,
                "corroborated": corroborated,
            }
        return result

    def _update_fast_averages(self, indicators):
        for name, indicator in indicators.items():
            score = indicator["combined_deviation"]
            if score is None or indicator.get("status") in {
                "Keyboard-only change", "Mouse-only change", "Insufficient data",
            }:
                self.rising_reports[name] = 0
                continue
            previous = self.fast_averages.get(name)
            current = score if previous is None else (
                (1.0 - BETA) * previous + BETA * score
            )
            rising = previous is not None and current > previous
            self.fast_averages[name] = current
            count = self.rising_reports.get(name, 0) + 1 if rising else 0
            self.rising_reports[name] = count
        self.baseline_frozen = any(count >= 3 for count in self.rising_reports.values())

    def _update_baseline(self, features, cluster_id, z_scores, indicators, sufficient):
        normal = all(item["status"] == "Normal" for item in indicators.values())
        mad = float(np.mean([abs(value) for value in z_scores.values()])) \
            if z_scores else math.inf
        if not sufficient or mad >= 1.0 or not normal or self.baseline_frozen:
            return False
        stats = self.mode_stats[str(cluster_id)]
        for key, raw in features.items():
            if raw is None or key not in stats:
                continue
            stat = stats[key]
            old_mean = float(stat["mean"])
            delta = float(raw) - old_mean
            stat["variance"] = max(
                (1.0 - ALPHA) * float(stat["variance"]) + ALPHA * delta * delta,
                0.0,
            )
            stat["mean"] = (1.0 - ALPHA) * old_mean + ALPHA * float(raw)
            stat["count"] += 1
        return True

    def analyze(self, features, meta, window_seconds=60.0, sufficient=True,
                allow_learning=True):
        signature = interaction_vector(features, meta, window_seconds)
        if self.centroids is None:
            if sufficient and allow_learning:
                self.observations.append({
                    "signature": signature.tolist(),
                    "features": {key: value for key, value in features.items()
                                 if value is None or isinstance(value, (int, float))},
                })
                self.observations = self.observations[-MAX_MODE_OBSERVATIONS:]
                if len(self.observations) >= MIN_MODE_WINDOWS:
                    self._fit_modes()
            if allow_learning:
                self.save()
            return {
                "mode": None,
                "mode_status": "provisional",
                "insufficient": True,
                "indicators": self._score_indicators({}),
                "z_scores": {},
            }

        cluster_id, mode, distance, familiar = self._mode_for(signature)
        stats = self.mode_stats.get(str(cluster_id), {})
        mode_ready = self.mode_window_counts.get(str(cluster_id), 0) \
            >= MIN_BASELINE_WINDOWS_PER_MODE
        mode_calibration_added = False
        if sufficient and familiar and not mode_ready and allow_learning:
            self._add_mode_calibration(cluster_id, features)
            mode_calibration_added = True
            mode_ready = self.mode_window_counts[str(cluster_id)] \
                >= MIN_BASELINE_WINDOWS_PER_MODE
        z_scores = (self._z_scores(features, cluster_id)
                    if sufficient and familiar and mode_ready else {})
        indicators = self._score_indicators(z_scores)
        device_data_available = all(
            indicator["keyboard_deviation"] is not None
            and indicator["mouse_deviation"] is not None
            for indicator in indicators.values()
        )
        valid = sufficient and familiar and mode_ready and device_data_available
        updated = False
        if allow_learning:
            if valid:
                self._update_fast_averages(indicators)
                updated = self._update_baseline(
                    features, cluster_id, z_scores, indicators, valid,
                )
            if mode_calibration_added or valid:
                self.save()
        return {
            "mode": mode if familiar else None,
            "mode_status": ("unfamiliar" if not familiar else
                            "known" if mode_ready else "provisional"),
            "mode_distance": distance,
            "insufficient": not valid,
            "z_scores": z_scores,
            "indicators": indicators,
            "baseline_updated": updated,
            "baseline_frozen": self.baseline_frozen,
        }
