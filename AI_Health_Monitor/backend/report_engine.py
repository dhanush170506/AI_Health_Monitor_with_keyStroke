"""
AI Health Monitor - Report Engine (Module 10)
=============================================

Every two 1-minute analyses it builds a report averaging the last two windows,
adds behavioural changes vs. the personal baseline plus a non-medical
recommendation, and appends it to data/reports.json (atomic write).
"""

import json
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


def build_report(window_results, vector, deviations, baseline_summary, now_epoch=None):
    """Build one report dict from the last two 1-minute analysis results."""
    if not window_results:
        return None
    now_epoch = now_epoch if now_epoch is not None else time.time()
    recent = window_results[-REPORT_INTERVAL_WINDOWS:]
    n = len(recent)

    def avg(key):
        return int(round(sum(r[key] for r in recent) / n)) if n else 0

    rep = {
        "start_time": _fmt_ts(now_epoch - 60.0 * (n - 1)),
        "end_time": _fmt_ts(now_epoch),
        "avg_fatigue": avg("fatigue_score"),
        "avg_stress": avg("stress_score"),
        "avg_anxiety": avg("anxiety_score"),
        "avg_motor_risk": avg("motor_risk_score"),
        "overall_status": band(max(avg("fatigue_score"), avg("stress_score"),
                                   avg("anxiety_score"), avg("motor_risk_score"))),
        "behavioural_changes": behavioural_changes(vector, deviations, baseline_summary),
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
