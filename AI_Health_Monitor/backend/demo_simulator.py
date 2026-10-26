"""
AI Health Monitor - Demo Scenario Simulator (Module 12)
=======================================================

Generates synthetic RAW keyboard/mouse event streams that resemble each
behavioural condition. The events are pushed through the EXACT same pipeline
as live data (feature extraction -> baseline deviation -> ML model -> fusion
-> report engine). Dashboard numbers are never set directly.

UNITS (must match the frontend and feature engine):
  * all timestamps in MILLISECONDS
  * mouse coordinates in px inside a virtual 800x500 activity area

Scenarios: normal | fatigue | stress | anxiety | motor
"""

import math
import random

DURATION_MS = 60_000.0
AREA_W, AREA_H = 800.0, 500.0


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def _jit(base, spread):
    return base * (1.0 + random.uniform(-spread, spread))


# Per-scenario behavioural parameter profiles (human-plausible values).
# err   = typing error rate (% of typed chars reported as mistakes)
# tp    = (probability, lo_s, hi_s) of a mid-typing thinking pause (seconds)
SCENARIOS = {
    #                       dwell(ms)        flight(ms)        wpm   bs%
    "normal":  dict(dwell=(105, 22), flight=(145, 40), wpm=50, bs=6,
                    err=2.0, tp=(0.35, 0.6, 1.6),
                    # mouse: target-seeking with modest noise
                    mspeed=310, mnoise=0.10, pause=(0.3, 1.2),
                    mstd=0.35, clicks=13),
    "fatigue": dict(dwell=(150, 34), flight=(235, 70), wpm=36, bs=7,
                    err=3.5, tp=(0.55, 1.4, 3.4),
                    mspeed=195, mnoise=0.14, pause=(1.2, 3.2),
                    mstd=0.45, clicks=9),
    "stress":  dict(dwell=(88, 18), flight=(112, 42), wpm=63, bs=13,
                    err=9.0, tp=(0.12, 0.45, 1.1),
                    mspeed=440, mnoise=0.16, pause=(0.15, 0.45),
                    mstd=0.55, clicks=23),
    "anxiety": dict(dwell=(112, 46), flight=(165, 90), wpm=46, bs=11,
                    err=6.5, tp=(0.50, 0.5, 2.2),
                    mspeed=360, mnoise=0.34, pause=(0.2, 0.9),
                    mstd=0.85, clicks=20),
    "motor":   dict(dwell=(125, 45), flight=(185, 110), wpm=42, bs=9,
                    err=5.0, tp=(0.60, 1.5, 3.8),
                    mspeed=240, mnoise=0.42, pause=(1.0, 3.0),
                    mstd=0.90, clicks=8),
}


def error_counts(scenario, n_events, severity=1.0):
    """PRIVACY-SAFE aggregate pair (n_errors, n_chars) for a simulated window.

    Mirrors what the assessment frontend reports for real users: counts only,
    never the characters themselves. Used by BOTH training and demo mode so
    the error_rate feature has a consistent meaning everywhere.
    """
    p = _blend_params(scenario, severity)
    return max(0, int(round(n_events * p["err"] / 100.0))), max(0, int(n_events))


def _blend_params(scenario, severity):
    """Interpolate a scenario's parameters towards the 'normal' profile.

    severity 1.0 = full condition, 0.0 = indistinguishable from normal.
    Real behaviour is a continuum, so training windows use random severities
    (which makes classes genuinely overlap) while demo scenarios run at 1.0.
    """
    p = SCENARIOS[scenario]
    if scenario == "normal" or severity >= 1.0:
        return p
    base = SCENARIOS["normal"]
    blended = {}
    for k, v in p.items():
        b = base[k]
        if isinstance(v, tuple):
            blended[k] = tuple(bv + (x - bv) * severity for bv, x in zip(b, v))
        elif isinstance(v, (int, float)):
            blended[k] = b + (v - b) * severity
        else:
            blended[k] = v
    return blended


def generate_keyboard(scenario, duration_ms=DURATION_MS, severity=1.0):
    """Simulate raw key timing events in ms (no characters - timing only).

    Includes a slow within-minute drift (people speed up and slow down in
    waves) plus occasional thinking pauses, so windows are realistically
    noisy rather than robotically uniform."""
    p = _blend_params(scenario, severity)
    wpm0 = _jit(p["wpm"], 0.15)
    n_keys = int(wpm0 * 5.0 / 60.0 * (duration_ms / 1000.0))
    bs_prob = p["bs"] / 100.0
    pause_prob, pause_lo_s, pause_hi_s = p["tp"]

    events = []
    t = random.uniform(0.0, 1500.0)
    dm, ds = p["dwell"]
    fm, fs = p["flight"]
    drift_phase = random.uniform(0, 2 * math.pi)
    for i in range(n_keys):
        progress = t / duration_ms
        drift = 1.0 + 0.10 * math.sin(2 * math.pi * progress + drift_phase)
        dwell = _clip(random.gauss(dm * drift * _jit(1.0, 0.12), ds), 35.0, 900.0)
        events.append({"t_down": round(t, 1),
                       "t_up": round(t + dwell, 1),
                       "is_backspace": random.random() < bs_prob})
        flight = _clip(random.gauss(fm * (2.0 - drift) * _jit(1.0, 0.15), fs),
                       25.0, 8000.0)
        t += dwell + flight
        if random.random() < pause_prob:           # scenario-dependent thinking pause
            t += random.uniform(pause_lo_s, pause_hi_s) * 1000.0
        if t > duration_ms:
            break
    return events


def generate_mouse(scenario, duration_ms=DURATION_MS, severity=1.0):
    """Target-seeking mouse: moves to a goal point (the 'target'), clicks it,
    pauses, then picks the next one. Mirrors the assessment page's mouse
    task. Produces realistic speeds, efficiency, jitter and pause patterns.

    Returns (events, acquisition_times_ms) where acquisition time is the
    duration of each click-terminated burst (move to target + click)."""
    p = _blend_params(scenario, severity)
    x, y = random.uniform(60, AREA_W - 60), random.uniform(60, AREA_H - 60)
    base_speed = _jit(p["mspeed"], 0.15)   # px/s
    pause_min, pause_max = p["pause"]

    events = []
    acquisitions = []
    t = 0.0
    target_clicks = max(1, int(_jit(p["clicks"], 0.25)))
    n_bursts = target_clicks + max(1, int(target_clicks * 0.4))  # + non-click moves
    click_flags = [i < target_clicks for i in range(n_bursts)]
    random.shuffle(click_flags)

    def new_goal():
        return (random.uniform(40, AREA_W - 40), random.uniform(40, AREA_H - 40))

    gx, gy = new_goal()

    for burst in range(n_bursts):
        if t >= duration_ms:
            break
        speed_now = base_speed * _jit(1.0, p["mstd"])   # per-burst speed variation
        burst_start = t
        will_click = click_flags[burst]
        clicked = False
        while t < duration_ms:
            dt = random.uniform(35.0, 110.0)             # ms per sample
            # steer towards goal + scenario noise
            desired = math.atan2(gy - y, gx - x)
            heading = desired + random.gauss(0.0, p["mnoise"])
            # occasionally overshoot/turn sharply (jitter source)
            if random.random() < p["mnoise"] * 0.5:
                heading += random.choice([-1, 1]) * random.uniform(1.2, 2.4)
            step = speed_now * (dt / 1000.0)
            x += math.cos(heading) * step
            y += math.sin(heading) * step
            x = _clip(x, 1.0, AREA_W - 1.0)
            y = _clip(y, 1.0, AREA_H - 1.0)
            t += dt
            near_goal = math.hypot(gx - x, gy - y) < 28.0
            do_click = will_click and near_goal
            events.append({"x": round(x, 2), "y": round(y, 2),
                           "t": round(t, 1), "clicked": do_click})
            if do_click:
                clicked = True
                acquisitions.append(t - burst_start)     # target acquired
                break
            if near_goal:                                # reached without clicking
                break
        # -- pause between targets (no samples -> detected as pause) --
        t += random.uniform(pause_min, pause_max) * 1000.0
        gx, gy = new_goal()
    return events, acquisitions


def generate(scenario, duration_ms=DURATION_MS, offset_ms=0.0, severity=1.0):
    """Return (key_events, mouse_events, acquisition_times_ms) for a window."""
    if scenario not in SCENARIOS:
        scenario = "normal"
    keys = generate_keyboard(scenario, duration_ms, severity)
    mouse, acquisitions = generate_mouse(scenario, duration_ms, severity)
    if offset_ms:
        for e in keys:
            e["t_down"] = round(e["t_down"] + offset_ms, 1)
            e["t_up"] = round(e["t_up"] + offset_ms, 1)
        for e in mouse:
            e["t"] = round(e["t"] + offset_ms, 1)
    return keys, mouse, acquisitions
