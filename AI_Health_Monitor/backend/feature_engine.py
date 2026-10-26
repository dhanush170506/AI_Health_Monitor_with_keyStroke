"""
AI Health Monitor - Behavioural Feature Engine
==============================================

Extracts the 12 behavioural features expected by the trained RandomForest model
from raw keyboard / mouse event streams. All math is defensive: missing data,
empty windows, missing key-ups, and duplicate mouse positions never crash the
engine and never produce NaN / Infinity in the output vector.

PRIVACY: this module never sees the *content* of what was typed. The frontend
strips key characters before transmission; only timing metadata arrives here.

Feature vector (order matters - must match model/feature_names.json):

    1.  typing_speed_wpm          words per minute estimated from inter-key rhythm
    2.  avg_dwell_time            mean key press duration (ms)
    3.  dwell_std                 std-dev of key press durations (ms)
    4.  avg_flight_time           mean gap between one key-up and the next key-down (ms)
    5.  flight_std                std-dev of flight times (ms)
    6.  backspace_rate            backspace-ish corrections per 100 key-down events
    7.  error_rate                mistyped chars per 100 typed chars (client aggregate)
    8.  typing_pause_duration     mean mid-typing pause length (ms)
    9.  mouse_speed               mean cursor speed (px/s)
    10. mouse_speed_std           std-dev of cursor speed (px/s)
    11. mouse_acceleration        mean |speed change| between movement samples (px/s^2)
    12. mouse_pause_duration      mean pause length between movement bursts (ms)
    13. mouse_jitter              normalized rapid-direction-change measure (0..1)
    14. mouse_efficiency          straight-line displacement / path length (0..1)
    15. click_rate                clicks per minute
    16. target_acquisition_time   mean time to reach+click a target (ms, 0 = n/a)
"""

import math

import numpy as np

# Canonical feature order - the trained model and predictor both use this list.
FEATURE_NAMES = [
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

# ---------------------------------------------------------------------------
# UNIT CONVENTION: every timestamp in the pipeline is in MILLISECONDS
# (frontend sends performance.now()-based ms; demo simulator emits ms).
# Durations (dwell/flight/pause) therefore come out in ms, speeds in px/s.
# ---------------------------------------------------------------------------

# Hard upper bounds used for sanity-clamping (absorbs clock glitches).
MAX_DWELL_MS = 10_000.0     # a held key longer than 10 s is treated as an outlier
MAX_FLIGHT_MS = 60_000.0    # > 60 s between keys is a pause, not typing rhythm
MAX_SPEED_PXS = 20_000.0    # px/s speed cap (teleporting cursors)
MAX_ACCEL_PX2 = 500_000.0   # px/s^2 cap (teleporting cursors)
MAX_TARGET_MS = 30_000.0    # > 30 s to acquire one target is an outlier


# ---------------------------------------------------------------------------
# Preprocessing / cleaning (Module 3)
# ---------------------------------------------------------------------------

def clean_keyboard_events(events, window_duration=None):
    """Sanitize a list of keyboard timing events.

    Accepts dicts with: t_down (float), t_up (float or None),
    is_backspace (bool). Content fields, if present, are ignored entirely.
    """
    if not events:
        return []
    cleaned = []
    for ev in events:
        try:
            t_down = float(ev.get("t_down"))
            t_up = ev.get("t_up")
            t_up = float(t_up) if t_up is not None else None
        except (TypeError, ValueError):
            continue
        # invalid timestamps
        if not math.isfinite(t_down) or t_down < 0:
            continue
        if t_up is not None:
            if not math.isfinite(t_up) or t_up < t_down:
                t_up = None  # missing / reversed key-up -> treat as missing
        if t_up is not None and (t_up - t_down) > MAX_DWELL_MS:
            t_up = t_down + MAX_DWELL_MS  # clamp pathological dwell
        cleaned.append({
            "t_down": t_down,
            "t_up": t_up,
            "is_backspace": bool(ev.get("is_backspace", False)),
        })
    cleaned.sort(key=lambda e: e["t_down"])
    return cleaned


def clean_mouse_events(events):
    """Sanitize mouse movement/click samples: {x, y, t, clicked(bool)}."""
    if not events:
        return []
    cleaned = []
    for ev in events:
        try:
            x = float(ev.get("x"))
            y = float(ev.get("y"))
            t = float(ev.get("t"))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(t)) or t < 0:
            continue
        cleaned.append({"x": x, "y": y, "t": t,
                        "clicked": bool(ev.get("clicked", False))})
    cleaned.sort(key=lambda e: e["t"])
    # Drop consecutive duplicate positions (zero-distance samples add no signal
    # and distort pause detection).
    deduped = []
    last = None
    for ev in cleaned:
        if last is not None and abs(ev["x"] - last["x"]) < 1e-9 \
                and abs(ev["y"] - last["y"]) < 1e-9:
            # keep the later timestamp for click bookkeeping purposes
            last["clicked"] = last["clicked"] or ev["clicked"]
            continue
        deduped.append(ev)
        last = ev
    return deduped


# ---------------------------------------------------------------------------
# Keyboard features (Module 4)
# ---------------------------------------------------------------------------

def _std(values):
    if len(values) < 2:
        return 0.0
    return float(np.std(values))


def keyboard_features(events, window_duration=60.0, n_chars_typed=None,
                      n_errors=0, target_acquisitions=None):
    """Compute keyboard features from cleaned events. Returns dict of floats.

    n_chars_typed / n_errors are PRIVACY-SAFE aggregate counters produced by
    the client (the actual characters never leave the browser).
    target_acquisitions is a list of target-acquisition durations (ms) from
    the mouse-target task; 0.0 means no acquisitions were reported.
    """
    events = clean_keyboard_events(events)
    n_downs = len(events)
    dwells = [e["t_up"] - e["t_down"] for e in events if e["t_up"] is not None]

    # Flight times: next key-down minus previous key-up (requires key-ups).
    flights = []
    prev_up = None
    for e in events:
        if prev_up is not None:
            gap = e["t_down"] - prev_up
            if 0 <= gap <= MAX_FLIGHT_MS:
                flights.append(gap)
        if e["t_up"] is not None:
            prev_up = e["t_up"]

    avg_dwell = float(np.mean(dwells)) if dwells else 0.0
    avg_flight = float(np.mean(flights)) if flights else 0.0

    # Typing speed (WPM): keystrokes-per-minute / average word length.
    # Use observed typing span (first to last key-down) when available so that
    # short bursts of typing are not diluted by idle time. Timestamps are ms.
    if n_downs >= 2:
        span_s = max((events[-1]["t_down"] - events[0]["t_down"]) / 1000.0, 1.0)
    else:
        span_s = max(window_duration, 1.0)
    kpm = n_downs / span_s * 60.0
    wpm = kpm / 5.0  # 5 chars per word, standard approximation

    backspaces = sum(1 for e in events if e["is_backspace"])
    backspace_rate = (backspaces / n_downs * 100.0) if n_downs else 0.0

    # Error rate: client-reported wrong characters per 100 typed characters.
    # Falls back to 0.0 when the client did not report (plain free typing).
    error_rate = 0.0
    if n_chars_typed and n_chars_typed > 0:
        error_rate = float(n_errors) / float(n_chars_typed) * 100.0

    # Typing pauses: flight gaps longer than 400 ms are hesitations, not rhythm.
    PAUSE_GAP_MS = 400.0
    typing_pauses = [g for g in flights if g >= PAUSE_GAP_MS]
    typing_pause_mean = float(np.mean(typing_pauses)) if typing_pauses else 0.0

    acquisition_mean = (float(np.mean(target_acquisitions))
                        if target_acquisitions else 0.0)

    return {
        "typing_speed_wpm": _safe(wpm, lo=0.0, hi=250.0),
        "avg_dwell_time": _safe(avg_dwell, lo=0.0, hi=2000.0),
        "dwell_std": _safe(_std(dwells), lo=0.0, hi=2000.0),
        "avg_flight_time": _safe(avg_flight, lo=0.0, hi=10000.0),
        "flight_std": _safe(_std(flights), lo=0.0, hi=10000.0),
        "backspace_rate": _safe(backspace_rate, lo=0.0, hi=100.0),
        "error_rate": _safe(error_rate, lo=0.0, hi=100.0),
        "typing_pause_duration": _safe(typing_pause_mean, lo=0.0, hi=30000.0),
        "target_acquisition_time": _safe(acquisition_mean, lo=0.0, hi=MAX_TARGET_MS),
        "n_key_events": n_downs,
    }


# ---------------------------------------------------------------------------
# Mouse features (Module 4)
# ---------------------------------------------------------------------------

def mouse_features(events, window_duration=60.0):
    """Compute mouse features from cleaned events. Returns dict of floats."""
    events = clean_mouse_events(events)
    n = len(events)
    clicks = sum(1 for e in events if e["clicked"])

    speeds = []
    path_len = 0.0
    pauses = []           # durations (ms) of gaps where the cursor was still
    dir_changes = 0       # jitter signal: count of sharp heading changes
    accels = []           # |velocity change| / dt (px/s^2)
    burst_efficiencies = []    # straightness (displacement/path) of each burst

    PAUSE_THRESHOLD_MS = 500.0
    JITTER_ANGLE_RAD = math.radians(55)   # a heading change sharper than this...
    MIN_SEG_MS = 5.0                      # ...on a segment this short = jitter

    # small mutable holders so the nested closure can update them
    burst_path = [0.0]
    burst_first = [None, None]
    burst_last = [None, None]

    def close_burst():
        if burst_path[0] > 1e-6 and burst_first[0] is not None:
            straight = math.hypot(burst_last[0] - burst_first[0],
                                  burst_last[1] - burst_first[1])
            burst_efficiencies.append(min(1.0, straight / burst_path[0]))

    for i in range(1, n):
        a, b = events[i - 1], events[i]
        dt = b["t"] - a["t"]                      # ms
        if dt <= 0:
            continue
        dist = math.hypot(b["x"] - a["x"], b["y"] - a["y"])
        path_len += dist
        if dt > PAUSE_THRESHOLD_MS:
            pauses.append(dt)
            close_burst()
            burst_path[0] = 0.0
            burst_first[0] = None
        else:
            speed = (dist / dt * 1000.0) if dist > 0 else 0.0   # px/s
            speeds.append(min(speed, MAX_SPEED_PXS))
            burst_path[0] += dist
            if burst_first[0] is None:
                burst_first[0], burst_first[1] = a["x"], a["y"]
            burst_last[0], burst_last[1] = b["x"], b["y"]

        # heading change between consecutive segments (jitter detector)
        # acceleration accumulates in the same pass (|dv|/dt)
        if i >= 2 and dt > 1e-6:
            c = events[i - 2]
            dt_prev = a["t"] - c["t"]
            d_prev = math.hypot(a["x"] - c["x"], a["y"] - c["y"])
            if dt_prev > 1e-6 and d_prev > 1e-6:
                v1x, v1y = (a["x"] - c["x"]) / dt_prev * 1000.0, (a["y"] - c["y"]) / dt_prev * 1000.0
                v2x, v2y = (b["x"] - a["x"]) / dt * 1000.0, (b["y"] - a["y"]) / dt * 1000.0
                accel = math.hypot(v2x - v1x, v2y - v1y) / (dt / 1000.0)
                accels.append(min(accel, MAX_ACCEL_PX2))

        if i >= 2:
            c = events[i - 2]
            dt_prev = a["t"] - c["t"]
            d_prev = math.hypot(a["x"] - c["x"], a["y"] - c["y"])
            if dt_prev >= MIN_SEG_MS and dt >= MIN_SEG_MS and d_prev > 1e-6 and dist > 1e-6:
                v1 = ((a["x"] - c["x"]) / dt_prev, (a["y"] - c["y"]) / dt_prev)
                v2 = ((b["x"] - a["x"]) / dt, (b["y"] - a["y"]) / dt)
                dot = v1[0] * v2[0] + v1[1] * v2[1]
                norm = math.hypot(*v1) * math.hypot(*v2)
                if norm > 1e-9:
                    cosang = max(-1.0, min(1.0, dot / norm))
                    angle = math.acos(cosang)
                    if angle >= JITTER_ANGLE_RAD:
                        dir_changes += 1
    close_burst()

    span_s = ((events[-1]["t"] - events[0]["t"]) / 1000.0) if n >= 2 \
        else max(window_duration, 1.0)
    span_s = max(span_s, 1e-6)

    mean_speed = float(np.mean(speeds)) if speeds else 0.0
    std_speed = _std(speeds)

    # Jitter: fraction of movement segments containing a sharp direction
    # change, scaled into a 0..1 measure calibrated so normal target-seeking
    # mouse motion lands near the training-data normal (~0.03).
    segments = max(n - 1, 1)
    jitter = min(1.0, (dir_changes / segments) * 0.25)

    # Movement efficiency: mean straightness (displacement/path) of individual
    # movement bursts - robust to users naturally wandering around the area.
    efficiency = (float(np.mean(burst_efficiencies))
                  if burst_efficiencies else 0.0)

    click_rate = clicks / (span_s / 60.0) if span_s > 0 else 0.0

    avg_pause = float(np.mean(pauses)) if pauses else 0.0

    mean_accel = float(np.mean(accels)) if accels else 0.0

    return {
        "mouse_speed": _safe(mean_speed, lo=0.0, hi=MAX_SPEED_PXS),
        "mouse_speed_std": _safe(std_speed, lo=0.0, hi=MAX_SPEED_PXS),
        "mouse_acceleration": _safe(mean_accel, lo=0.0, hi=MAX_ACCEL_PX2),
        "mouse_pause_duration": _safe(avg_pause, lo=0.0, hi=10000.0),
        "mouse_jitter": _safe(jitter, lo=0.0, hi=1.0),
        "mouse_efficiency": _safe(efficiency, lo=0.0, hi=1.0),
        "click_rate": _safe(click_rate, lo=0.0, hi=600.0),
        "n_mouse_points": n,
        "n_clicks": clicks,
    }


# ---------------------------------------------------------------------------
# Full vector
# ---------------------------------------------------------------------------

def extract_features(key_events, mouse_events, window_duration=60.0,
                     n_chars_typed=None, n_errors=0, target_acquisitions=None):
    """Extract the full 16-feature vector as an ordered dict of plain floats.

    Always returns a complete, finite vector (safe defaults when a modality is
    silent). Also returns sample-count metadata so callers can judge whether
    the window contained enough behaviour to analyze.
    """
    kf = keyboard_features(key_events, window_duration, n_chars_typed=n_chars_typed,
                           n_errors=n_errors, target_acquisitions=target_acquisitions)
    mf = mouse_features(mouse_events, window_duration)

    vector = {
        "typing_speed_wpm": kf["typing_speed_wpm"],
        "avg_dwell_time": kf["avg_dwell_time"],
        "dwell_std": kf["dwell_std"],
        "avg_flight_time": kf["avg_flight_time"],
        "flight_std": kf["flight_std"],
        "backspace_rate": kf["backspace_rate"],
        "error_rate": kf["error_rate"],
        "typing_pause_duration": kf["typing_pause_duration"],
        "mouse_speed": mf["mouse_speed"],
        "mouse_speed_std": mf["mouse_speed_std"],
        "mouse_acceleration": mf["mouse_acceleration"],
        "mouse_pause_duration": mf["mouse_pause_duration"],
        "mouse_jitter": mf["mouse_jitter"],
        "mouse_efficiency": mf["mouse_efficiency"],
        "click_rate": mf["click_rate"],
        "target_acquisition_time": kf["target_acquisition_time"],
    }

    # Final NaN/Infinity guard - the vector must always be model-safe.
    for k, v in vector.items():
        if not math.isfinite(v):
            vector[k] = 0.0

    meta = {
        "n_key_events": kf["n_key_events"],
        "n_mouse_points": mf["n_mouse_points"],
        "n_clicks": mf["n_clicks"],
    }
    return vector, meta


def enough_data(meta):
    """Decide whether a window's behaviour is meaningful enough to analyze.

    Hackathon rule of thumb: some keyboard activity OR some mouse movement.
    """
    return meta["n_key_events"] >= 5 or meta["n_mouse_points"] >= 10


def _safe(value, lo, hi):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return lo
    if not math.isfinite(v):
        return lo
    return max(lo, min(hi, v))
