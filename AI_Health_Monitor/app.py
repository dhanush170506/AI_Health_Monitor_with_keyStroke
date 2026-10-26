"""
AI Health Monitor - Flask Backend (Modules 2, 9, 12, 13, 14, 15)
================================================================

Wires together feature extraction, personal baseline, the trained model,
motor-risk anomaly scoring, score fusion and the 2-minute report engine.

PRIVACY: the backend only ever receives timing metadata (no key characters,
no text content) and mouse coordinates inside the designated activity area.

Run:  python app.py   ->  http://127.0.0.1:5000
"""

import math
import logging
import os
import threading
import time

from flask import Flask, jsonify, request, send_from_directory

from backend.analyzer import analyze, band_label
from backend.baseline import Baseline
from backend.behavioural_analysis import BehaviouralAnalysis
from backend.feature_engine import (extract_features, enough_data,
                                    clean_keyboard_events, clean_mouse_events)
from backend.predictor import Predictor
from backend import report_engine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
BASELINE_PATH = os.path.join(DATA_DIR, "baseline.json")
REPORTS_PATH = os.path.join(DATA_DIR, "reports.json")
BEHAVIOURAL_BASELINE_PATH = os.path.join(DATA_DIR, "behavioural_baseline.json")

WINDOW_SECONDS = 60.0          # Module 9: 1-minute live analysis
REPORT_EVERY_WINDOWS = 2       # Module 10: report per 2 minutes
CALIBRATION_SECONDS = 120.0    # Module 5: default calibration (~2 minutes)

app = Flask(__name__, static_folder=None)
LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session state (single monitor, single user - hackathon scope)
# ---------------------------------------------------------------------------

class Monitor:
    def __init__(self):
        self.lock = threading.RLock()
        self.predictor = Predictor()                      # model or None
        self.baseline = Baseline(BASELINE_PATH, CALIBRATION_SECONDS)
        self.baseline.load()                              # persist across restarts
        self.behavioural = BehaviouralAnalysis(BEHAVIOURAL_BASELINE_PATH)

        self.reset_session(keep_baseline=True)

        # aggregates reported by the assessment page (counts only - privacy)
        self.n_chars_typed = 0
        self.n_errors = 0
        self.target_acquisitions = []   # list of ms durations

        # 1-minute analyzer loop
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # -- session lifecycle ---------------------------------------------------

    def reset_session(self, keep_baseline=True):
        with self.lock:
            self.active = False
            self.session_start = None
            self.next_window_at = None
            self.window_index = 0
            self.valid_window_count = 0
            self.key_events = []      # [{t_down, t_up, is_backspace}] (seconds)
            self.mouse_events = []    # [{x, y, t, clicked}]
            self.history = []         # per-minute analysis results
            self.reports = report_engine.load_reports(REPORTS_PATH)
            self.last_result = None
            self.last_insufficient = False
            self.demo_scenario = None
            self.demo_runs = 0
            self.n_chars_typed = 0
            self.n_errors = 0
            self.target_acquisitions = []
            if not keep_baseline:
                self.baseline.reset()
                self.behavioural.reset()

    def start(self):
        with self.lock:
            if self.active:
                return
            self.active = True
            self.session_start = time.time()
            self.next_window_at = self.session_start + WINDOW_SECONDS
            self.calibrating = not self.baseline.established
            self.calibration_started = self.session_start
            self.demo_scenario = None

    def stop(self):
        with self.lock:
            self.active = False
            # preserve reports & history; stop accepting events

    # -- event ingestion (Module 2) -------------------------------------------

    def add_events(self, key_events, mouse_events, n_chars_typed=0, n_errors=0,
                   target_acquisitions=None):
        """Append sanitized client events. Client sends milliseconds relative to
        its own session clock - only relative timing is used.

        n_chars_typed / n_errors are PRIVACY-SAFE aggregate counts (the typed
        characters themselves never leave the browser); target_acquisitions is
        a list of durations in ms, not coordinates or content."""
        with self.lock:
            if not self.active:
                return
            self.key_events.extend(clean_keyboard_events(key_events))
            self.mouse_events.extend(clean_mouse_events(mouse_events))
            try:
                self.n_chars_typed += max(0, int(n_chars_typed))
                self.n_errors += max(0, int(n_errors))
            except (TypeError, ValueError):
                pass
            if isinstance(target_acquisitions, list):
                for a in target_acquisitions:
                    try:
                        v = float(a)
                        if math.isfinite(v) and 0.0 <= v <= 60000.0:
                            self.target_acquisitions.append(v)
                    except (TypeError, ValueError):
                        continue

    # -- the 1-minute pipeline (Module 9) --------------------------------------

    def _loop(self):
        while True:
            time.sleep(1.0)
            try:
                with self.lock:
                    run = (self.active and self.session_start is not None
                           and self.next_window_at is not None
                           and time.time() >= self.next_window_at)
                if run:
                    self.process_window()
                    with self.lock:
                        # schedule the next 1-minute boundary; if the machine
                        # slept or lagged, skip missed boundaries (no burst)
                        now = time.time()
                        if now >= self.next_window_at + WINDOW_SECONDS:
                            self.next_window_at = now + WINDOW_SECONDS
                        else:
                            self.next_window_at += WINDOW_SECONDS
            except Exception:
                # Module 15: the monitor loop must never die
                LOGGER.exception("Minute analysis loop failed")

    def process_window(self, demo=False):
        """Analyze the events accumulated in the current window."""
        with self.lock:
            keys = self.key_events
            mouse = self.mouse_events
            n_chars = self.n_chars_typed
            n_errs = self.n_errors
            acq = list(self.target_acquisitions)
            self.key_events = []
            self.mouse_events = []
            self.n_chars_typed = 0
            self.n_errors = 0
            self.target_acquisitions = []

        vector, meta = extract_features(keys, mouse, WINDOW_SECONDS,
                                        n_chars_typed=n_chars or None,
                                        n_errors=n_errs,
                                        target_acquisitions=acq or None)
        sufficient = enough_data(meta)
        with self.lock:
            behavioural = self.behavioural.analyze(
                vector, meta, WINDOW_SECONDS, sufficient=sufficient,
            )
        now = time.time()

        with self.lock:
            if self.active and not demo:
                self.window_index += 1

        # --- calibration phase (Module 5) ---
        if self.baseline.established is False and not demo:
            if sufficient:
                self.baseline.add_calibration_sample(vector)
            with self.lock:
                self.last_insufficient = not sufficient
                if not sufficient:
                    self.last_result = None
                calib_left = max(0.0, self.baseline.calibration_seconds
                                 - (now - self.calibration_started))
                self.calibration_remaining = calib_left
            return

        # --- normal analysis ---
        deviations = (self.baseline.deviations(vector)
                      if self.baseline.established else
                      {k: 0.0 for k in vector})
        result = analyze(vector, deviations, self.predictor,
                         self.baseline.established)

        with self.lock:
            if not sufficient and not demo:
                # Do not leave a previous window's scores displayed as current.
                self.last_insufficient = True
                self.last_result = None
                return
            self.last_insufficient = False
            result["timestamp"] = report_engine._fmt_ts(now)
            result["window"] = self.window_index
            result["vector"] = {
                k: round(v, 3) if v is not None else None
                for k, v in vector.items()
            }
            result["deviations"] = {k: round(v, 3) for k, v in deviations.items()}
            result["behavioral_analysis"] = behavioural
            result["valid"] = True
            self.last_result = result
            self.history.append(result)
            self.valid_window_count += 1
            if len(self.history) > 180:
                self.history = self.history[-180:]

            # Module 5: gentle incremental baseline update on normal-ish windows
            if behavioural.get("baseline_updated"):
                self.baseline.update_incremental(vector, rate=0.02)

            # Module 10: 2-minute report
            if self.valid_window_count % REPORT_EVERY_WINDOWS == 0:
                report_windows = [
                    item for item in self.history
                    if item.get("valid") and not item.get("demo_scenario")
                ][-REPORT_EVERY_WINDOWS:]
                rep = report_engine.build_report(
                    report_windows, vector, deviations,
                    self.baseline.summary())
                self.reports = report_engine.append_report(rep, REPORTS_PATH)

    # -- demo mode (Module 12) ---------------------------------------------------

    def _demo_vector(self, scenario, windows=3):
        """Simulate `windows` separate 60-second windows of a scenario and
        average their feature vectors.

        Each window is extracted individually with the SAME extract_features()
        used by live monitoring - keeping every feature in the exact
        distribution shape the model was trained on (concatenating windows
        into one stream would create cross-window timing artifacts)."""
        from backend import demo_simulator
        from backend.feature_engine import FEATURE_NAMES
        vecs = []
        for _ in range(windows):
            keys, mouse, acq = demo_simulator.generate(
                scenario, duration_ms=WINDOW_SECONDS * 1000.0)
            n_err, n_chars = demo_simulator.error_counts(scenario, n_events=len(keys))
            vec, _meta = extract_features(keys, mouse, WINDOW_SECONDS,
                                          n_chars_typed=n_chars or None,
                                          n_errors=n_err,
                                          target_acquisitions=acq or None)
            vecs.append(vec)
        return {f: sum(v[f] for v in vecs) / len(vecs) for f in FEATURE_NAMES}

    def run_demo(self, scenario):
        """Generate synthetic RAW events for a scenario and push them through
        the exact same pipeline (features -> baseline -> model -> fusion ->
        report). Nothing here manipulates dashboard numbers directly."""
        from backend import demo_simulator
        with self.lock:
            if not self.active:
                self.start()
            # For an instant reviewer experience: if no personal baseline exists
            # yet, seed it with three simulated NORMAL sessions (each an average
            # of three 60 s windows - the same shape demo scenarios use), so
            # demo deviations are meaningful and stable.
            if not self.baseline.established:
                for _ in range(3):
                    self.baseline.add_calibration_sample(self._demo_vector("normal"))
            self.demo_scenario = scenario
            self.demo_runs += 1

        # Simulate a ~3-minute collection period (3 independent 1-minute
        # windows of the scenario, averaged) for stable demo results.
        vector = self._demo_vector(scenario)
        meta = {"n_key_events": 999, "n_mouse_points": 999, "n_clicks": 999}
        deviations = (self.baseline.deviations(vector)
                      if self.baseline.established else
                      {k: 0.0 for k in vector})
        result = analyze(vector, deviations, self.predictor,
                         self.baseline.established)
        now = time.time()
        with self.lock:
            self.last_insufficient = False
            result["timestamp"] = report_engine._fmt_ts(now)
            result["window"] = f"demo-{self.demo_runs}"
            result["vector"] = {
                k: round(v, 3) if v is not None else None
                for k, v in vector.items()
            }
            result["deviations"] = {k: round(v, 3) for k, v in deviations.items()}
            result["behavioral_analysis"] = self.behavioural.analyze(
                vector, meta, WINDOW_SECONDS, sufficient=True,
                allow_learning=False,
            )
            result["demo_scenario"] = scenario
            self.last_result = result
            self.history.append(result)
            if len(self.history) > 180:
                self.history = self.history[-180:]
            # demo: a report after every scenario run (uses last 2 windows)
            rep = report_engine.build_report(
                [result], vector,
                deviations, self.baseline.summary())
            self.reports = report_engine.append_report(rep, REPORTS_PATH)
        return result

    # -- API payloads -------------------------------------------------------------

    def status(self):
        with self.lock:
            elapsed = (time.time() - self.session_start) if self.active else 0.0
            return {
                "monitoring_active": self.active,
                "elapsed_seconds": int(elapsed),
                "model_ready": self.predictor.ready,
                "baseline": self.baseline.summary(),
                "behavioral_baseline": self.behavioural.summary(),
                "calibrating": bool(
                    self.active and (
                        not self.baseline.established
                        or not self.behavioural.summary()["established"]
                    )
                ),
                "calibration_remaining": int(max(0.0, self.baseline.calibration_seconds
                                                 - (time.time() - self.calibration_started)))
                if self.active and not self.baseline.established else 0,
                "window_index": self.window_index,
                "demo_scenario": self.demo_scenario,
                "demo_runs": self.demo_runs,
                "buffer": {
                    "key_events": len(self.key_events),
                    "mouse_points": len(self.mouse_events),
                    "chars_typed": self.n_chars_typed,
                    "targets_hit": len(self.target_acquisitions),
                },
            }

    def live(self):
        st = self.status()
        with self.lock:
            last = self.last_result
            history = list(self.history)
            reports = list(self.reports)
        trend = {
            "labels": [h.get("window", i + 1) for i, h in enumerate(history)],
            "fatigue": [
                h.get("behavioral_analysis", {}).get("indicators", {})
                .get("fatigue", {}).get("score") for h in history
            ],
            "stress": [
                h.get("behavioral_analysis", {}).get("indicators", {})
                .get("stress_tension", {}).get("score") for h in history
            ],
            "anxiety": [None for _ in history],
            "motor": [
                h.get("behavioral_analysis", {}).get("indicators", {})
                .get("motor_consistency", {}).get("score") for h in history
            ],
        }
        return {
            **st,
            "last_result": last,
            "insufficient": bool(self.last_insufficient
                                 and not (last and last.get("demo_scenario"))),
            "collecting_message": "Collecting more behavioural data...",
            "features": (last or {}).get("vector"),
            "trend": trend,
            "reports": reports[-10:][::-1],   # newest first
        }


MONITOR = Monitor()


# ---------------------------------------------------------------------------
# Routes (Module 13)
# ---------------------------------------------------------------------------

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/assessment")
def assessment():
    """Monkeytype-style behavioural assessment page (opened in a new tab by
    the dashboard's START MONITORING button)."""
    return send_from_directory(FRONTEND_DIR, "assessment.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


@app.route("/api/sentences")
def api_sentences():
    """A random set of assessment sentences (Module 18). The bank is neutral
    content; the user's typed answers never reach this server."""
    from backend.sentence_bank import random_sentences
    try:
        n = int(request.args.get("n", 10))
    except (TypeError, ValueError):
        n = 10
    return jsonify({"sentences": random_sentences(n)})


@app.route("/api/status")
def api_status():
    return jsonify(MONITOR.status())


@app.route("/api/events", methods=["POST"])
def api_events():
    data = request.get_json(silent=True) or {}
    keys = data.get("keyboard") or []
    mouse = data.get("mouse") or []
    # Privacy: only timing fields and aggregate counts are ever read here. Any
    # content fields present in the payload are ignored and never stored.
    MONITOR.add_events(keys, mouse,
                       n_chars_typed=data.get("n_chars_typed") or 0,
                       n_errors=data.get("n_errors") or 0,
                       target_acquisitions=data.get("target_acquisitions"))
    return jsonify({"ok": True, "buffered": MONITOR.status()["buffer"]})


@app.route("/api/live")
def api_live():
    return jsonify(MONITOR.live())


@app.route("/api/reports")
def api_reports():
    with MONITOR.lock:
        reports = list(MONITOR.reports)
    return jsonify({"reports": reports[::-1]})


@app.route("/api/start", methods=["POST"])
def api_start():
    MONITOR.start()
    return jsonify({"ok": True, **MONITOR.status()})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    MONITOR.stop()
    return jsonify({"ok": True, **MONITOR.status()})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    data = request.get_json(silent=True) or {}
    keep = data.get("keep_baseline", True)
    MONITOR.reset_session(keep_baseline=bool(keep))
    return jsonify({"ok": True, **MONITOR.status()})


@app.route("/api/demo", methods=["POST"])
def api_demo():
    data = request.get_json(silent=True) or {}
    scenario = str(data.get("scenario", "normal")).lower()
    if scenario not in ("normal", "fatigue", "stress", "anxiety", "motor"):
        return jsonify({"error": "unknown scenario"}), 400
    result = MONITOR.run_demo(scenario)
    return jsonify({"ok": True, "result": result, **MONITOR.status()})


@app.errorhandler(404)
def not_found(_e):
    return jsonify({"error": "not found"}), 404


@app.errorhandler(500)
def server_error(_e):
    return jsonify({"error": "internal error"}), 500


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        port = int(os.environ.get("PORT", 5000))
    except (TypeError, ValueError):
        port = 5000
    if port <= 0:
        port = 5000  # PORT=0 would make Flask pick a random port
    print("=" * 64)
    print("  AI HEALTH MONITORING SYSTEM - behavioural screening prototype")
    print("  NOT a medical diagnostic system.")
    print(f"  Model ready: {MONITOR.predictor.ready}")
    print(f"  Open http://127.0.0.1:{port} in your browser.")
    print("=" * 64)
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
