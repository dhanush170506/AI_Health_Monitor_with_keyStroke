# Run doc — AI Health Monitor (Flask)

## Reproduce artifacts (fresh checkout)

All steps run from `AI_Health_Monitor/` with Python 3.10+ on PATH:

1. Install dependencies (includes `flask-cors` — verify it imports before starting):
   ```
   pip install -r requirements.txt
   ```
2. The trained model is committed at `model/health_model.pkl` (plus
   `model/feature_names.json`). If it is missing, retrain:
   ```
   python train_model.py
   ```
3. `data/reports.json` is created/maintained by the app itself; no seed needed.
   `data/baseline.json` is optional — it is created on first demo run / calibration.

## Run the server

```
cd AI_Health_Monitor
python app.py            # serves http://127.0.0.1:5000
```

- Default port is 5000; override with `PORT` env var (values <= 0 fall back to 5000).
- The Flask app is NOT an npm project. For the detached preview launch, use
  PowerShell with the real executable `python.exe` (not a shim), `app.py` as the
  argument, working directory `AI_Health_Monitor`, and stdout/stderr redirected
  to DIFFERENT log files under `.freebuff/`.
- Confirm liveness with `Get-Process -Id <pid>` and
  `curl http://127.0.0.1:5000/api/status` before registering the preview.
