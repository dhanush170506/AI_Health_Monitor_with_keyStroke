# AI-Based Health Monitoring System Using Keystroke Dynamics and Mouse Movement Analysis

A live, non-intrusive **behavioural health monitoring prototype** built for a 5-hour
hackathon. It observes how you type and move your mouse (never *what* you type),
extracts behavioural features, compares them against your **personal baseline**,
runs a **trained Random Forest model**, and continuously reports estimated
**fatigue, stress, anxiety, and a motor-symptom risk indicator**.

> **Honesty & medical disclaimer**
> This is an AI **behavioural screening prototype trained on synthetic behavioural
> data**. It is **NOT clinically validated** and is **NOT a medical diagnostic
> system**. The "Motor Symptom Risk Indicator" does **not** diagnose Parkinson's
> disease or any other condition. Scores are personalized behavioural deviation
> indicators only.

---

## 1. Project Overview

- Monitors keyboard **timing dynamics** (dwell, flight, rhythm, corrections) and
  mouse **movement dynamics** (speed, pauses, jitter, efficiency, clicks).
- Runs a **1-minute live analysis loop**: features → baseline comparison →
  ML prediction → anomaly scoring → fused health scores.
- Generates a **2-minute report** averaging the last two windows, with
  behavioural changes vs. baseline and a non-medical recommendation.
- **Monkeytype-style assessment flow:** clicking **Start Monitoring** opens the
  **Behavioural Health Assessment** page in a new tab — 10 random sentences
  (from a 74-sentence bank) with real-time character highlighting plus a
  mouse-target clicking task. After sentence 10 there is an **Assessment
  Complete** summary and a **Continue Monitoring** free-interaction mode.
- Includes a **Demo Mode** (NORMAL / FATIGUE / STRESS / ANXIETY / MOTOR RISK) so
  reviewers can see predictions respond instantly — simulated raw events go
  through the *exact same pipeline* as live data.
- **Privacy by design:** typed characters never leave the browser; only timing
  metadata, aggregate counters (chars typed, errors, target times) and in-area
  mouse coordinates are transmitted and stored.

## 2. Architecture

```
KEYBOARD + MOUSE (assessment tab: timing-only listeners + aggregate counters)
        ↓  (batched metadata, no key characters, no text)
Flask /api/events      Flask /api/sentences (random 10 of 74-sentence bank)
        ↓
FEATURE EXTRACTION  (backend/feature_engine.py — 16 features)
        ↓
PERSONAL BASELINE   (backend/baseline.py — z-score deviations)
        ↓
TRAINED ML MODEL    (backend/predictor.py — RandomForest predict_proba)
        +                                    +
ANOMALY DETECTION   (backend/analyzer.py — motor-risk deviation score)
        ↓
SCORE FUSION        (backend/predictor.py analyze())
        ↓
1-MINUTE LIVE ANALYSIS      2-MINUTE REPORT ENGINE
(backend loop in app.py)    (backend/report_engine.py → data/reports.json)
        ↓
DASHBOARD (frontend/index.html — polling)     ASSESSMENT (frontend/assessment.html — capture + tasks)
```

### File map

```
AI_Health_Monitor/
├── app.py                  Flask app, session management, 1-min loop, APIs
├── train_model.py          Synthetic data generation + model training
├── requirements.txt
├── README.md
├── model/
│   ├── health_model.pkl    Trained RandomForest (created by train_model.py)
│   ├── feature_names.json  Canonical feature order for predictor.py
│   └── training_report.json
├── backend/
│   ├── feature_engine.py   Preprocessing + 16-feature extraction
│   ├── baseline.py         Personal baseline (calibration, z-scores, persistence)
│   ├── predictor.py        Model loading, predict_proba, score fusion
│   ├── analyzer.py         Motor-risk anomaly score + analyze() entry point
│   ├── report_engine.py    2-minute report building + data/reports.json
│   ├── sentence_bank.py    74 neutral sentences → random 10 per assessment
│   └── demo_simulator.py   Raw-event generators for the demo scenarios
├── frontend/
│   ├── index.html          Dashboard (analysis + reports + demo)
│   ├── assessment.html     Behavioural Health Assessment (new tab)
│   ├── style.css           Styling
│   ├── assessment.css      Assessment-specific styling
│   ├── monitor.js          Dashboard rendering/polling (no capture)
│   └── assessment.js       Typing test, mouse targets, timing-only capture
└── data/
    ├── baseline.json       Persisted personal baseline (auto-created)
    └── reports.json        Persisted 2-minute reports (auto-created)
```

## 3. Features

- Live typing + mouse behavioural capture inside the app (no system-wide hooks).
- 16 behavioural features: typing speed (WPM), avg dwell time, dwell std,
  avg flight time, flight std, backspace rate, error rate, typing pause
  duration, mouse speed, mouse speed std, mouse acceleration, mouse pause
  duration, mouse jitter, movement efficiency, click rate, target acquisition
  time.
- Defensive preprocessing: missing key-ups, duplicate mouse points, invalid
  timestamps, and empty windows never crash the pipeline or produce NaN/Infinity.
- Personal baseline with configurable calibration (~2 min default), persisted to
  `data/baseline.json`, with incremental updates that ignore anomalous windows.
- Motor Symptom Risk Indicator: statistical deviation score (0–100) across
  dwell variability, flight-time slowing, mouse jitter, movement efficiency,
  and mouse-speed variation — with human-readable indicator strings.
- Fused 0–100 scores for Fatigue / Stress / Anxiety + Overall status bands.
- 2-minute reports with behavioural changes and recommendations, persisted.
- Dashboard: health cards, live metrics, baseline panel, SVG trend chart,
  latest report, report history, demo buttons, typing + mouse areas.

## 4. Installation (Windows)

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 5. How to Train the Model

```bat
python train_model.py
```

Prints dataset size, train/test accuracy, classification report and confusion
matrix, then saves `model/health_model.pkl`, `model/feature_names.json` and
`model/training_report.json`. (A pre-trained model ships with the repo; retrain
any time.)

## 6. How to Run

```bat
python app.py
```

Then open **http://127.0.0.1:5000** in your browser (URL is printed in the console).

## 7. How to Start Monitoring (the assessment flow)

1. On the dashboard click **▶ Start Monitoring** — a new tab opens:
   **Behavioural Health Assessment** (`/assessment`). The dashboard stays live
   in its own tab.
2. The assessment draws **10 random sentences** (10–16 words each, different
   every session). Type each sentence exactly as shown — characters highlight
   green (correct) / red (wrong) in real time, mistakes can be fixed with
   Backspace, and finishing a sentence auto-advances. A small status panel
   logs WPM / dwell / flight / errors per completed sentence.
3. In the **Mouse Activity Area**, click the highlighted targets as they appear
   (acquisition time is measured automatically).
4. First session: a **CALIBRATING PERSONAL BASELINE** banner shows for ~2
   minutes of assessment activity, then switches to **PERSONAL BASELINE
   ESTABLISHED**.
5. Every 60 s the analysis updates on both tabs (the assessment shows a compact
   "Current behavioural state" panel); every 120 s a new **2-Minute Report**
   appears on the dashboard.
6. After sentence 10: an **Assessment Complete** summary (total time, average
   WPM/dwell/flight/error-rate, targets completed) → **Return to Dashboard** or
   **Continue Monitoring** (free typing + mouse area, same 1-min/2-min cycle).
7. **Stop** preserves reports; **Reset Session** clears the current session but
   keeps the personal baseline.

## 8. Demo Instructions (for reviewers)

- Click any demo button — **NORMAL / FATIGUE / STRESS / ANXIETY / MOTOR RISK**.
- The backend simulates ~3 minutes of raw keyboard/mouse events for that
  condition and pushes them through the identical pipeline
  (feature extraction → baseline → model → fusion → report engine).
  No dashboard numbers are set directly.
- Expect roughly: NORMAL → all scores low; FATIGUE → high fatigue with
  dwell/flight indicators; STRESS → high stress with jitter/efficiency
  indicators; ANXIETY → high anxiety; MOTOR RISK → elevated motor score with
  motor indicators. The 2-minute report and trend chart update too.
- If no personal baseline exists yet, one is seeded automatically from simulated
  normal windows so the demo always works immediately.

## 9. Privacy Design

- The JavaScript key handlers read **only** `performance.now()` timestamps and a
  backspace/delete flag. The key value is never read, sent, or stored —
  passwords and prose cannot leak because they are never transmitted.
- The assessment page compares typed characters against the target sentence
  **locally in the browser** for highlighting; what is transmitted is limited
  to timestamps, a backspace flag, and aggregate counters (chars typed, error
  count, target-acquisition durations).
- Mouse coordinates are captured **only inside the designated activity areas**.
- The backend stores only timing metadata, aggregate counts, coordinates, and
  derived features/scores in local JSON files. No cloud, no database server.
- The UI states it plainly: *"Typed content is never stored. Only behavioural
  timing and movement features are analyzed."*

## 10. ML Methodology (honest version)

- **No clinical dataset exists for this hackathon**, so training data is
  **synthetic**: `backend/demo_simulator.py` generates human-plausible raw
  keyboard/mouse event windows (per-virtual-user offsets, within-minute drift,
  severity blending between normal and each condition), which are passed through
  the **same `extract_features()` pipeline used live**, then labelled
  NORMAL / FATIGUE / STRESS / ANXIETY by the generating scenario.
- A `RandomForestClassifier` (200 trees, depth 12, balanced class weights) is
  trained on ~1,440 windows with a stratified 75/25 split. Because severities
  blend towards normal, the classes genuinely overlap — test accuracy sits in
  the high-90s rather than a suspicious 100%.
- **Fusion:** model `predict_proba` is the primary signal; personal-baseline
  z-scores act as a multiplicative confidence adjustment; the motor-risk score
  is a separate statistical deviation indicator that also informs overall status.
- All scores are 0–100 with bands 0–29 LOW/NORMAL, 30–59 MODERATE, 60–100 ELEVATED.

## 11. Limitations

- Not clinically validated; not a diagnostic or medical device.
- Trained on synthetic behavioural data; real-world generalization is unknown.
- Single-user, single-browser-tab monitoring; typing capture works when the app
  tab has focus (no OS-level keylogging — by design).
- One-minute windows with sparse activity return "Collecting more behavioural
  data…" rather than fake precision.
- Baseline needs a few minutes of real usage before deviations are meaningful.
- Behavioural states are inferred from timing proxies only; mood/pain/etc. are
  not measured.

## 12. Hackathon Explanation

Built in a 5-hour hackathon window as an end-to-end MVP: a working Flask
backend, a genuinely trained scikit-learn model, a defensible personal-baseline
mechanism, a polished reviewer-ready dashboard, and an instant demo mode —
prioritizing *runs without errors → type → mouse → features change → model
loaded → baseline → live scores → reports → demo → polish*, in that order.

## API Summary

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Dashboard |
| `/assessment` | GET | Behavioural Health Assessment page |
| `/api/sentences` | GET | Random assessment sentences (`?n=10`) |
| `/api/status` | GET | Monitoring state, baseline, buffers |
| `/api/events` | POST | Keyboard/mouse metadata + aggregate counters (timing only) |
| `/api/live` | GET | Latest analysis, features, trend, reports |
| `/api/reports` | GET | All persisted 2-minute reports |
| `/api/start` | POST | Start monitoring / calibration |
| `/api/stop` | POST | Stop accepting events (reports preserved) |
| `/api/reset` | POST | Reset session (`{"keep_baseline": true}` default) |
| `/api/demo` | POST | `{"scenario": "normal\|fatigue\|stress\|anxiety\|motor"}` |
