"""
AI Health Monitor - Report Engine (Module 10)
=============================================

Every two 1-minute analyses it builds a report averaging the last two windows,
adds behavioural changes vs. the personal baseline plus a non-medical
recommendation, and appends it to data/reports.json (atomic write).
"""

import json
import math
import os
import time
from datetime import datetime, timezone

REPORT_INTERVAL_WINDOWS = 2   # one report per 2 one-minute windows


def _fmt_ts(epoch):
    return datetime.fromtimestamp(epoch, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def band(v):
    v = max(0, min(100, int(v)))
    return "NORMAL" if v < 30 else ("MODERATE" if v < 60 else "ELEVATED")


def behavioural_changes(vector, deviations, baseline_summary):
    """Human-readable change lines, e.g. 'Typing speed decreased 12% ...'."""
    changes = []
    if not baseline_summary or not baseline_summary.get("established"):
        return ["No personal baseline yet - changes are relative to this session."]

    mean = baseline_summary["mean"]

    def pct(key):
        base = mean.get(key, 0.0)
        if base < 1e-9:
            return None
        return (float(vector.get(key, 0.0)) - base) / base * 100.0

    def add(text, value, threshold=8.0):
        if value is None:
            return
        if abs(value) >= threshold:
            sign = "increased" if value > 0 else "decreased"
            changes.append(text.format(sign=sign, value=abs(int(round(value)))))

    add("Typing speed {sign} {value}% from baseline", pct("typing_speed_wpm"))
    add("Dwell time {sign} {value}%", pct("avg_dwell_time"))
    add("Flight time {sign} {value}%", pct("avg_flight_time"))
    add("Backspace rate {sign} {value}%", pct("backspace_rate"), threshold=25.0)
    add("Mouse speed {sign} {value}%", pct("mouse_speed"))
    add("Mouse jitter {sign} {value}%", pct("mouse_jitter"), threshold=20.0)
    add("Click rate {sign} {value}%", pct("click_rate"), threshold=25.0)

    eff_base = mean.get("mouse_efficiency", 0.0)
    if eff_base > 1e-9:
        eff_delta = float(vector.get("mouse_efficiency", 0.0)) - eff_base
        if abs(eff_delta) >= 0.08:
            direction = "improved" if eff_delta > 0 else "reduced"
            changes.append(f"Movement efficiency {direction} "
                           f"({abs(eff_delta) * 100:.0f} pts on the 0-100 scale)")

    # stable lines for near-baseline features (nice for demo narrative)
    if not changes:
        changes.append("All tracked behaviours remained close to baseline "
                       "(e.g. mouse jitter remained stable).")
    return changes


def _recommendation_for(report):
    scores = [report["avg_fatigue"], report["avg_stress"],
              report["avg_anxiety"], report["avg_motor_risk"]]
    scores = [score for score in scores if score is not None]
    if not scores:
        return "Insufficient valid activity windows to summarize."
    if report["overall_status"] == "NORMAL":
        return ("Your recent behaviour stayed close to your personal baseline. "
                "Keep going comfortably.")
    if report["overall_status"] == "MODERATE":
        if report["avg_fatigue"] == max(scores):
            return ("Consider taking a short break if you have been working "
                    "continuously.")
        return ("A short pause and a stretch may help; some behavioural signals "
                "are moderately elevated versus your baseline.")
    return ("Take a proper break and resume at a comfortable pace. Behavioural "
            "screening only - not a medical assessment.")


def _summarize_indicators(window_results):
    summaries = {}
    names = ("fatigue", "stress_tension", "motor_consistency")
    for name in names:
        rows = [
            result.get("behavioral_analysis", {}).get("indicators", {}).get(name, {})
            for result in window_results
        ]
        scores = [float(row["score"]) for row in rows
                  if row.get("score") is not None
                  and row.get("corroborated") is True]
        if scores and len(scores) == len(window_results):
            score = int(round(sum(scores) / len(scores)))
            status = ("Normal" if score == 0 else
                      "Mild change" if score < 50 else "Notable change")
        else:
            score = None
            statuses = {row.get("status") for row in rows}
            if statuses == {"Keyboard-only change"}:
                status = "Keyboard-only change"
            elif statuses == {"Mouse-only change"}:
                status = "Mouse-only change"
            else:
                status = "Insufficient data"
        summaries[name] = {
            "score": score,
            "status": status,
            "corroborated_windows": len(scores),
            "valid_windows": len(window_results),
        }
    return summaries


def build_report(window_results, vector, deviations, baseline_summary, now_epoch=None):
    """Build one report dict from the last two 1-minute analysis results."""
    recent = [
        row for row in window_results[-REPORT_INTERVAL_WINDOWS:]
        if row.get("valid", True)
    ]
    if not recent:
        return None
    now_epoch = now_epoch if now_epoch is not None else time.time()
    n = len(recent)

    def avg(key):
        values = [
            float(row[key]) for row in recent
            if row.get(key) is not None
            and math.isfinite(float(row[key]))
        ]
        return int(round(sum(values) / len(values))) if values else None

    fatigue, stress = avg("fatigue_score"), avg("stress_score")
    anxiety, motor = avg("anxiety_score"), avg("motor_risk_score")
    values = [value for value in (fatigue, stress, anxiety, motor)
              if value is not None]
    behavioral_indicators = _summarize_indicators(recent)
    behavioral_scores = [
        item["score"] for item in behavioral_indicators.values()
        if item["score"] is not None
    ]
    if len(behavioral_scores) != len(behavioral_indicators):
        behavioral_status = "Insufficient data"
    else:
        highest = max(behavioral_scores)
        behavioral_status = (
            "Normal" if highest == 0 else
            "Mild change" if highest < 50 else "Notable change"
        )
    rep = {
        "start_time": _fmt_ts(now_epoch - 60.0 * (n - 1)),
        "end_time": _fmt_ts(now_epoch),
        "avg_fatigue": fatigue,
        "avg_stress": stress,
        "avg_anxiety": anxiety,
        "avg_motor_risk": motor,
        "overall_status": band(max(values)) if values else "INSUFFICIENT",
        "behavioral_status": behavioral_status,
        "behavioural_changes": behavioural_changes(vector, deviations, baseline_summary),
        "behavioral_indicators": behavioral_indicators,
    }
    rep["recommendation"] = _recommendation_for(rep)
    return rep


def load_reports(path="data/reports.json"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            reports = json.load(f)
        return reports if isinstance(reports, list) else []
    except (OSError, ValueError):
        return []


def append_report(report, path="data/reports.json"):
    """Append a report to data/reports.json (atomic write). Returns list."""
    reports = load_reports(path)
    reports.append(report)
    try:
        tmp = path + ".tmp"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2)
        os.replace(tmp, path)
    except OSError:
        pass
    return reports
