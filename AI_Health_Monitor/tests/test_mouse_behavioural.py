import math
import os
import tempfile
import threading
import time
import unittest
import json
import numpy as np

from backend.behavioural_analysis import BehaviouralAnalysis
from backend.feature_engine import (
    FEATURE_NAMES,
    enough_data,
    extract_features,
    mouse_features,
)
from backend.predictor import DEFAULT_FEATURE_NAMES
from backend.report_engine import build_report


def points(coords, interval=100, clicks=None):
    clicks = set(clicks or [])
    return [
        {"x": x, "y": y, "t": index * interval, "clicked": index in clicks}
        for index, (x, y) in enumerate(coords)
    ]


class MouseFeatureTests(unittest.TestCase):
    def test_straight_line_efficiency_is_one(self):
        features = mouse_features(points([(0, 0), (10, 0), (20, 0)]))
        self.assertAlmostEqual(features["mouse_path_efficiency"], 1.0)

    def test_curved_path_efficiency_is_below_one(self):
        features = mouse_features(points([(0, 0), (10, 0), (10, 10)]))
        self.assertLess(features["mouse_path_efficiency"], 1.0)

    def test_stationary_cursor_has_no_efficiency_or_division_error(self):
        features = mouse_features(points([(5, 5), (5, 5), (5, 5)]))
        self.assertIsNone(features["mouse_path_efficiency"])
        self.assertEqual(features["mouse_speed"], 0.0)

    def test_duplicate_timestamps_are_ignored(self):
        events = [{"x": 0, "y": 0, "t": 0}, {"x": 10, "y": 0, "t": 0},
                  {"x": 20, "y": 0, "t": 100}]
        features = mouse_features(events)
        self.assertEqual(features["mouse_path_length"], 10.0)
        self.assertEqual(features["mouse_average_velocity"], 100.0)

    def test_long_idle_gap_is_excluded_from_velocity_and_efficiency(self):
        events = [{"x": 0, "y": 0, "t": 0}, {"x": 10, "y": 0, "t": 100},
                  {"x": 20, "y": 0, "t": 10100},
                  {"x": 30, "y": 0, "t": 10200}]
        features = mouse_features(events)
        self.assertEqual(features["mouse_path_length"], 20.0)
        self.assertEqual(features["mouse_average_velocity"], 100.0)
        self.assertAlmostEqual(features["mouse_path_efficiency"], 1.0)

    def test_sudden_direction_change_is_counted(self):
        features = mouse_features(points([(0, 0), (10, 0), (10, 10)]))
        self.assertGreater(features["mouse_direction_changes"], 0)

    def test_smooth_movement_has_fewer_direction_changes(self):
        smooth = mouse_features(points([(0, 0), (10, 0), (20, 0), (30, 0)]))
        corner = mouse_features(points([(0, 0), (10, 0), (10, 10), (10, 20)]))
        self.assertLess(smooth["mouse_direction_changes"],
                        corner["mouse_direction_changes"])

    def test_jitter_is_deterministic(self):
        trajectory = points([(0, 0), (10, 0), (18, 3), (30, 0), (40, 0)])
        self.assertEqual(mouse_features(trajectory)["mouse_jitter_index"],
                         mouse_features(trajectory)["mouse_jitter_index"])

    def test_click_interval_mean_and_variability(self):
        events = points([(0, 0)] * 4, interval=100, clicks={0, 1, 3})
        features = mouse_features(events)
        self.assertEqual(features["mouse_click_interval_mean"], 150.0)
        self.assertAlmostEqual(features["mouse_click_interval_variability"], 1 / 3)

    def test_model_feature_order_is_preserved(self):
        with open(os.path.join("model", "feature_names.json"), encoding="utf-8") as stream:
            saved_order = json.load(stream)
        self.assertEqual(FEATURE_NAMES, DEFAULT_FEATURE_NAMES)
        self.assertEqual(FEATURE_NAMES, saved_order)

    def test_requested_mouse_metrics_are_in_feature_output(self):
        vector, _meta = extract_features(
            [], points([(0, 0), (10, 0), (20, 5)]),
        )
        expected = {
            "mouse_average_velocity", "mouse_median_velocity",
            "mouse_peak_velocity", "mouse_mean_absolute_acceleration",
            "mouse_acceleration_variability", "mouse_path_length",
            "mouse_path_efficiency", "mouse_direction_changes",
            "mouse_pause_duration", "mouse_pause_count", "mouse_jitter_index",
            "mouse_movement_smoothness", "mouse_click_interval_mean",
            "mouse_click_interval_variability",
        }
        self.assertTrue(expected.issubset(vector))

    def test_stationary_mouse_samples_are_insufficient_activity(self):
        self.assertFalse(enough_data({
            "n_key_events": 0,
            "n_mouse_points": 20,
            "n_mouse_movement_intervals": 0,
            "mouse_path_length": 0.0,
            "mouse_active_duration": 0.0,
        }))


class IndicatorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.analysis = BehaviouralAnalysis(
            os.path.join(self.temporary.name, "baseline.json")
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_zero_variance_produces_finite_z_score(self):
        self.analysis.mode_stats["0"] = {
            "mouse_jitter_index": {"count": 5, "mean": 0.0, "variance": 0.0}
        }
        scores = self.analysis._z_scores({"mouse_jitter_index": 1.0}, 0)
        self.assertEqual(scores["mouse_jitter_index"], 3.0)
        self.assertTrue(math.isfinite(scores["mouse_jitter_index"]))

    def test_z_scores_are_clipped_at_both_limits(self):
        self.analysis.mode_stats["0"] = {
            "x": {"count": 5, "mean": 0.0, "variance": 1.0}
        }
        high = self.analysis._z_scores({"x": 100.0}, 0)["x"]
        low = self.analysis._z_scores({"x": -100.0}, 0)["x"]
        self.assertEqual((high, low), (3.0, -3.0))

    def test_keyboard_only_change_is_not_corroborated(self):
        result = self.analysis._score_indicators({
            "backspace_rate": 2.0, "flight_std": 0.0,
            "mouse_jitter_index": 0.0, "mouse_direction_changes": 0.0,
            "mouse_click_interval_variability": 0.0,
        })["stress_tension"]
        self.assertEqual(result["status"], "Keyboard-only change")
        self.assertFalse(result["corroborated"])

    def test_mouse_only_change_is_not_corroborated(self):
        result = self.analysis._score_indicators({
            "backspace_rate": 0.0, "flight_std": 0.0,
            "mouse_jitter_index": 2.0, "mouse_direction_changes": 2.0,
            "mouse_click_interval_variability": 2.0,
        })["stress_tension"]
        self.assertEqual(result["status"], "Mouse-only change")
        self.assertFalse(result["corroborated"])

    def test_both_device_contributions_are_corroborated(self):
        result = self.analysis._score_indicators({
            "backspace_rate": 2.0, "flight_std": 2.0,
            "mouse_jitter_index": 2.0, "mouse_direction_changes": 2.0,
            "mouse_click_interval_variability": 2.0,
        })["stress_tension"]
        self.assertTrue(result["corroborated"])
        self.assertEqual(result["score"], 67)
        self.assertEqual(result["status"], "Notable change")

    def test_insufficient_activity_does_not_create_a_score(self):
        result = self.analysis.analyze({}, {}, sufficient=False)
        self.assertTrue(result["insufficient"])
        self.assertTrue(all(item["score"] is None
                            for item in result["indicators"].values()))

    def test_unfamiliar_interaction_mode_has_no_indicator_score(self):
        self.analysis.centroids = np.asarray([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [2.0, 2.0, 2.0, 0.0],
        ])
        self.analysis.acceptance_distance = 0.01
        self.analysis.cluster_labels = {"0": "typing-dominant",
                                        "1": "mixed",
                                        "2": "pointer-dominant"}
        self.analysis.mode_window_counts = {"0": 5, "1": 5, "2": 5}
        self.analysis.mode_stats = {"0": {}, "1": {}, "2": {}}
        result = self.analysis.analyze(
            {}, {"n_key_events": 1000000}, sufficient=True,
            allow_learning=False,
        )
        self.assertEqual(result["mode_status"], "unfamiliar")
        self.assertTrue(all(item["score"] is None
                            for item in result["indicators"].values()))

    def test_baseline_freezes_after_three_rising_reports(self):
        for score in (0.1, 0.2, 0.3, 0.4):
            self.analysis._update_fast_averages({
                "fatigue": {"combined_deviation": score}
            })
        self.assertTrue(self.analysis.baseline_frozen)

    def test_single_device_change_does_not_enter_fast_trend(self):
        self.analysis._update_fast_averages({
            "stress_tension": {
                "combined_deviation": 2.0,
                "status": "Keyboard-only change",
            }
        })
        self.assertNotIn("stress_tension", self.analysis.fast_averages)

    def test_guarded_baseline_uses_slow_preupdate_mean_variance(self):
        self.analysis.mode_stats["0"] = {
            "x": {"count": 5, "mean": 0.0, "variance": 1.0}
        }
        normal = {name: {"status": "Normal"} for name in
                  ("fatigue", "stress_tension", "motor_consistency")}
        updated = self.analysis._update_baseline(
            {"x": 0.5}, 0, {"x": 0.5}, normal, sufficient=True,
        )
        self.assertTrue(updated)
        self.assertAlmostEqual(self.analysis.mode_stats["0"]["x"]["mean"], 0.01)
        self.assertAlmostEqual(
            self.analysis.mode_stats["0"]["x"]["variance"], 0.985,
        )

    def test_kmeans_learns_three_modes_from_window_signatures(self):
        groups = ((60, 100, 1), (10, 600, 10), (35, 350, 5))
        for key_rate, path, clicks in groups:
            for _ in range(5):
                self.analysis.analyze(
                    {"typing_speed_wpm": key_rate, "mouse_path_length": path},
                    {"n_key_events": key_rate, "n_clicks": clicks,
                     "n_mouse_points": path},
                    sufficient=True,
                )
        self.assertTrue(self.analysis.summary()["established"])
        self.assertEqual(len(self.analysis.centroids), 3)
        self.assertEqual(self.analysis.observations, [])
        restored = BehaviouralAnalysis(self.analysis.path)
        self.assertTrue(restored.summary()["established"])
        self.assertEqual(restored.observations, [])


class ReportAndRetentionTests(unittest.TestCase):
    def test_report_ignores_invalid_windows(self):
        report = build_report([
            {"valid": False, "fatigue_score": 100, "stress_score": 100,
             "anxiety_score": 100, "motor_risk_score": 100},
            {"valid": True, "fatigue_score": 40, "stress_score": 20,
             "anxiety_score": 10, "motor_risk_score": 30},
        ], {}, {}, {"established": False}, now_epoch=1000)
        self.assertEqual(report["avg_fatigue"], 40)
        self.assertEqual(report["avg_stress"], 20)

    def test_report_does_not_treat_insufficient_indicator_as_zero(self):
        valid = {
            "valid": True,
            "fatigue_score": 20,
            "stress_score": 20,
            "anxiety_score": 20,
            "motor_risk_score": 20,
        }
        first = dict(valid, behavioral_analysis={
            "indicators": {
                "fatigue": {"score": 0, "corroborated": True, "status": "Normal"},
                "stress_tension": {"score": 0, "corroborated": True, "status": "Normal"},
                "motor_consistency": {"score": 0, "corroborated": True, "status": "Normal"},
            }
        })
        second = dict(valid, behavioral_analysis={
            "indicators": {
                "fatigue": {"score": None, "corroborated": False, "status": "Insufficient data"},
                "stress_tension": {"score": 0, "corroborated": True, "status": "Normal"},
                "motor_consistency": {"score": 0, "corroborated": True, "status": "Normal"},
            }
        })
        report = build_report([first, second], {}, {}, {}, now_epoch=1000)
        self.assertIsNone(report["behavioral_indicators"]["fatigue"]["score"])
        self.assertEqual(
            report["behavioral_indicators"]["fatigue"]["status"],
            "Insufficient data",
        )

    def test_window_extraction_discards_raw_trajectory_buffers(self):
        from app import Monitor

        class BaselineStub:
            established = False
            calibration_seconds = 120.0

            def add_calibration_sample(self, _vector):
                pass

        class BehaviourStub:
            def analyze(self, *_args, **_kwargs):
                return {"insufficient": True}

        monitor = Monitor.__new__(Monitor)
        monitor.lock = threading.RLock()
        monitor.key_events = []
        monitor.mouse_events = points([(0, 0), (10, 0), (20, 0)])
        monitor.n_chars_typed = 0
        monitor.n_errors = 0
        monitor.target_acquisitions = []
        monitor.baseline = BaselineStub()
        monitor.behavioural = BehaviourStub()
        monitor.active = False
        monitor.window_index = 0
        monitor.last_insufficient = False
        monitor.calibration_started = time.time()
        monitor.calibration_remaining = 0

        monitor.process_window()
        self.assertEqual(monitor.mouse_events, [])
        self.assertEqual(monitor.key_events, [])

    def test_live_endpoint_keeps_legacy_fields_and_adds_mode_status(self):
        from app import app

        response = app.test_client().get("/api/live")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("baseline", payload)
        self.assertIn("behavioral_baseline", payload)
        self.assertIn("features", payload)


if __name__ == "__main__":
    unittest.main()
