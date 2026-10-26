"""
AI Health Monitor - Behavioural Feature Engine
==============================================

Extracts the 16 behavioural features expected by the trained RandomForest model
plus additional mode-aware mouse measurements. All math is defensive: missing
data, empty windows, missing key-ups, and invalid mouse intervals never crash
the engine or create non-finite output.

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

The model's 16-feature order above is unchanged. Additional measurements
include mouse velocity median/peak (px/s), acceleration variability (px/s²),
path length (px), per-burst efficiency (0..1), direction changes per active
second, pause duration/count, conservative jitter index (0..1), dimensionless
movement smoothness (0..1), and click-interval mean/CV (ms / unitless).
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
MAX_MOUSE_GAP_MS = 500.0
MIN_MOVEMENT_PX = 1.0
PAUSE_SPEED_PXS = 2.0
MIN_PAUSE_MS = 250.0
MAX_PAUSE_INTERVAL_MS = 10_000.0
MIN_DIRECTION_ANGLE_RAD = math.radians(30.0)


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
    return cleaned


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
    """Compute mouse features from cleaned events.

    Time intervals are milliseconds, velocity is px/s, and acceleration is
    px/s². Optional estimates are ``None`` when there is insufficient data.
    """
    events = clean_mouse_events(events)
    clicks = sorted(e["t"] for e in events if e["clicked"])
    click_intervals = [b - a for a, b in zip(clicks, clicks[1:]) if b > a]
    speeds, vectors, pauses, segment_efficiencies = [], [], [], []
    movement_intervals = 0
    path_length = 0.0
    active_ms = 0.0
    segment = []

    def close_segment():
        if len(segment) >= 2:
            distances = [math.hypot(b[0] - a[0], b[1] - a[1])
                         for a, b in zip(segment, segment[1:])]
            length = sum(distances)
            if length >= MIN_MOVEMENT_PX:
                net = math.hypot(segment[-1][0] - segment[0][0],
                                 segment[-1][1] - segment[0][1])
                segment_efficiencies.append(max(0.0, min(1.0, net / length)))
        segment.clear()

    for a, b in zip(events, events[1:]):
        dt = b["t"] - a["t"]
        if dt <= 0:
            continue
        dx, dy = b["x"] - a["x"], b["y"] - a["y"]
        distance = math.hypot(dx, dy)
        if dt > MAX_MOUSE_GAP_MS:
            if MIN_PAUSE_MS <= dt <= MAX_PAUSE_INTERVAL_MS:
                pauses.append(dt)
            close_segment()
            continue
        if distance == 0:
            if dt >= MIN_PAUSE_MS:
                pauses.append(dt)
            continue

        speed = distance / (dt / 1000.0)
        if speed > MAX_SPEED_PXS:
            close_segment()
            continue
        path_length += distance
        active_ms += dt
        movement_intervals += 1
        speeds.append(speed)
        vectors.append((dx, dy, dt, distance, speed, a["t"], b["t"]))
        if not segment:
            segment.append((a["x"], a["y"]))
        segment.append((b["x"], b["y"]))
        if speed <= PAUSE_SPEED_PXS and dt >= MIN_PAUSE_MS:
            pauses.append(dt)
    close_segment()

    accelerations, legacy_accelerations, direction_changes = [], [], 0
    legacy_direction_changes = 0
    for prev, current in zip(vectors, vectors[1:]):
        if current[5] != prev[6]:
            continue
        prev_angle = math.atan2(prev[1], prev[0])
        angle = math.atan2(current[1], current[0])
        delta = (angle - prev_angle + math.pi) % (2 * math.pi) - math.pi
        if min(prev[3], current[3]) >= MIN_MOVEMENT_PX \
                and abs(delta) >= MIN_DIRECTION_ANGLE_RAD:
            direction_changes += 1
        acceleration = abs(current[4] - prev[4]) / (current[2] / 1000.0)
        if math.isfinite(acceleration) and acceleration <= MAX_ACCEL_PX2:
            accelerations.append(acceleration)
        previous_velocity = (prev[0] / (prev[2] / 1000.0),
                             prev[1] / (prev[2] / 1000.0))
        current_velocity = (current[0] / (current[2] / 1000.0),
                            current[1] / (current[2] / 1000.0))
        vector_acceleration = math.hypot(
            current_velocity[0] - previous_velocity[0],
            current_velocity[1] - previous_velocity[1],
        ) / (current[2] / 1000.0)
        if math.isfinite(vector_acceleration):
            legacy_accelerations.append(min(vector_acceleration, MAX_ACCEL_PX2))
        if min(prev[2], current[2]) >= 5.0:
            normalized_dot = math.cos(delta)
            legacy_angle = math.acos(max(-1.0, min(1.0, normalized_dot)))
            if legacy_angle >= math.radians(55.0):
                legacy_direction_changes += 1

    # Deterministic high-frequency residual proxy. A one-off bend is excluded:
    # only alternating short-scale curvature above pixel/relative noise floors
    # contributes to the estimate.
    residual = 0.0
    prior_turn = None
    for a, b, c in zip(events, events[1:], events[2:]):
        dt1, dt2 = b["t"] - a["t"], c["t"] - b["t"]
        if dt1 <= 0 or dt2 <= 0 or dt1 > MAX_MOUSE_GAP_MS \
                or dt2 > MAX_MOUSE_GAP_MS:
            prior_turn = None
            continue
        error = math.hypot(b["x"] - (a["x"] + c["x"]) / 2.0,
                           b["y"] - (a["y"] + c["y"]) / 2.0)
        dx1, dy1 = b["x"] - a["x"], b["y"] - a["y"]
        dx2, dy2 = c["x"] - b["x"], c["y"] - b["y"]
        turn = dx1 * dy2 - dy1 * dx2
        chord = math.hypot(c["x"] - a["x"], c["y"] - a["y"])
        if error >= max(MIN_MOVEMENT_PX, 0.05 * chord):
            if prior_turn is not None and turn * prior_turn < 0:
                residual += error
            prior_turn = turn
        else:
            prior_turn = None
    jitter = min(1.0, residual / path_length) if path_length >= MIN_MOVEMENT_PX else None
    efficiency = float(np.mean(segment_efficiencies)) if segment_efficiencies else None
    mean_speed = float(np.mean(speeds)) if speeds else None
    mean_accel = float(np.mean(accelerations)) if accelerations else None
    legacy_mean_accel = float(np.mean(legacy_accelerations)) \
        if legacy_accelerations else 0.0
    legacy_jitter = min(
        1.0, (legacy_direction_changes / max(len(events) - 1, 1)) * 0.25,
    )
    accel_variability = _std(accelerations) if len(accelerations) >= 2 else None
    smoothness = (1.0 / (1.0 + accel_variability / max(mean_accel, 1e-6))
                  if mean_accel is not None and accel_variability is not None else None)
    active_seconds = active_ms / 1000.0
    direction_rate = direction_changes / active_seconds if active_seconds > 0 else None
    click_mean = float(np.mean(click_intervals)) if click_intervals else None
    click_cv = (float(np.std(click_intervals)) / click_mean
                if len(click_intervals) >= 2 and click_mean > 0 else None)
    pause_total = sum(pauses)
    pause_ratio = (pause_total / (pause_total + active_ms)
                   if pause_total + active_ms else None)
    span_seconds = max((events[-1]["t"] - events[0]["t"]) / 1000.0, 1.0) \
        if len(events) >= 2 else max(window_duration, 1.0)
    net_displacement = (math.hypot(events[-1]["x"] - events[0]["x"],
                                   events[-1]["y"] - events[0]["y"])
                       if len(events) >= 2 else 0.0)

    return {
        "mouse_speed": _safe(mean_speed or 0.0, 0.0, MAX_SPEED_PXS),
        "mouse_speed_std": _safe(_std(speeds), 0.0, MAX_SPEED_PXS),
        "mouse_acceleration": _safe(legacy_mean_accel, 0.0, MAX_ACCEL_PX2),
        "mouse_pause_duration": _safe(pause_total / len(pauses) if pauses else 0.0,
                                      0.0, MAX_PAUSE_INTERVAL_MS),
        "mouse_jitter": _safe(legacy_jitter, 0.0, 1.0),
        "mouse_efficiency": _safe(efficiency, 0.0, 1.0) if efficiency is not None else 0.0,
        "click_rate": _safe(len(clicks) / (span_seconds / 60.0), 0.0, 600.0),
        "mouse_average_velocity": mean_speed,
        "mouse_median_velocity": float(np.median(speeds)) if speeds else None,
        "mouse_peak_velocity": max(speeds) if speeds else None,
        "mouse_mean_absolute_acceleration": mean_accel,
        "mouse_acceleration_variability": accel_variability,
        "mouse_path_length": path_length,
        "mouse_net_displacement": net_displacement,
        "mouse_path_efficiency": efficiency,
        "mouse_direction_changes": direction_rate,
        "mouse_pause_total_duration": pause_total,
        "mouse_pause_average_duration": (
            pause_total / len(pauses) if pauses else None
        ),
        "mouse_pause_count": len(pauses),
        "mouse_pause_duration_relative": pause_ratio,
        "mouse_jitter_index": jitter,
        "mouse_movement_smoothness": smoothness,
        "mouse_click_interval_mean": click_mean,
        "mouse_click_interval_variability": click_cv,
        "mouse_active_duration": active_seconds,
        "n_mouse_movement_intervals": movement_intervals,
        "n_mouse_points": len(events),
        "n_clicks": len(clicks),
    }


# ---------------------------------------------------------------------------
# Full vector
# ---------------------------------------------------------------------------

def extract_features(key_events, mouse_events, window_duration=60.0,
                     n_chars_typed=None, n_errors=0, target_acquisitions=None):
    """Extract the model feature vector plus derived mouse measurements.

    The canonical model fields remain first and unchanged. Optional mouse
    estimates are ``None`` when a reliable estimate cannot be made.
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
        **{k: v for k, v in mf.items() if k.startswith("mouse_")
           and k not in {"mouse_speed", "mouse_speed_std", "mouse_acceleration",
                         "mouse_pause_duration", "mouse_jitter",
                         "mouse_efficiency"}},
    }

    # Final NaN/Infinity guard - the vector must always be model-safe.
    for k, v in vector.items():
        if v is not None and not math.isfinite(v):
            vector[k] = 0.0

    meta = {
        "n_key_events": kf["n_key_events"],
        "n_mouse_points": mf["n_mouse_points"],
        "n_clicks": mf["n_clicks"],
        "n_mouse_movement_intervals": mf["n_mouse_movement_intervals"],
        "mouse_path_length": mf["mouse_path_length"],
        "mouse_active_duration": mf["mouse_active_duration"],
    }
    return vector, meta


def enough_data(meta):
    """Return true only when a window contains meaningful keyboard or motion."""
    keyboard = meta.get("n_key_events", 0) >= 5
    mouse = (
        meta.get("n_mouse_movement_intervals", 0) >= 3
        and meta.get("mouse_path_length", 0.0) >= 10.0
        and meta.get("mouse_active_duration", 0.0) >= 0.25
    )
    return keyboard or mouse


def _safe(value, lo, hi):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return lo
    if not math.isfinite(v):
        return lo
    return max(lo, min(hi, v))
