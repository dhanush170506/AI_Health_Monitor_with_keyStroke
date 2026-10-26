/*
 * AI Health Monitor - dashboard frontend logic
 * ============================================
 * PRIVACY: this page captures NO keystrokes and NO mouse movement. The
 * Behavioural Health Assessment tab (opened by Start Monitoring) does the
 * capture and sends only timing metadata + aggregate counters. This page
 * polls the analysis results and renders them.
 */

"use strict";

const API = {
  start: "/api/start",
  stop: "/api/stop",
  reset: "/api/reset",
  live: "/api/live",
  demo: "/api/demo",
  assessment: "/assessment",
};

const state = {
  pollTimer: null,
};

const $ = (id) => document.getElementById(id);

async function post(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!res.ok) throw new Error("POST failed: " + path);
  return res.json();
}

// ---------------------------------------------------------------------------
// Session controls
// ---------------------------------------------------------------------------

async function startMonitoring() {
  // The Start button's job: make sure the server session is running, then
  // open the Behavioural Health Assessment in a NEW TAB (window.open).
  try {
    await post(API.start);
    state.assessWindow = window.open(API.assessment, "_blank", "noopener");
    if (!state.assessWindow) {
      // popup blocked: fall back to same-tab navigation so the flow still works
      $("assess-hint").textContent =
        "Popup blocked — open the assessment via the button below, or allow popups for this site.";
    }
    $("assess-hint").textContent =
      "Assessment opened in a new tab. Complete the 10 sentences & mouse targets there — this dashboard updates live.";
    // immediate visual feedback (poll re-confirms within 2 s)
    $("status-pill").className = "status-pill active";
    $("status-text").textContent = "● Monitoring Active";
    updateButtons(true);
  } catch (err) {
    console.error("start failed", err);
  }
}

function updateButtons(active) {
  $("btn-start").disabled = !!active;
  $("btn-stop").disabled = !active;
}

async function stopMonitoring() {
  try {
    await post(API.stop);
    $("status-pill").className = "status-pill idle";
    $("status-text").textContent = "● Monitoring Stopped";
    updateButtons(false);
  } catch (err) {
    console.error("stop failed", err);
  }
}

async function resetSession() {
  try {
    await post(API.reset, { keep_baseline: true });
    $("status-pill").className = "status-pill idle";
    $("status-text").textContent = "● Idle";
    updateButtons(false);
    $("calibration-done").classList.add("hidden");
    $("calibration-banner").classList.add("hidden");
  } catch (err) {
    console.error("reset failed", err);
  }
}

async function runDemo(scenario) {
  const btns = document.querySelectorAll(".btn.demo");
  btns.forEach((b) => b.classList.add("busy"));
  $("demo-status").textContent = "Simulating " + scenario + " session (~3 min of behaviour)…";
  try {
    await post(API.demo, { scenario });
    $("demo-status").textContent =
      "Demo '" + scenario + "' complete — analysis and report updated.";
  } catch (err) {
    console.error("demo failed", err);
    $("demo-status").textContent = "Demo failed — is the backend running?";
  } finally {
    btns.forEach((b) => b.classList.remove("busy"));
  }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function fmtClock(sec) {
  const m = Math.floor(sec / 60), s = sec % 60;
  return String(m).padStart(2, "0") + ":" + String(s).padStart(2, "0");
}

function fmtNum(v, digits) {
  if (v === null || v === undefined || isNaN(v)) return "–";
  return Number(v).toFixed(digits);
}

function setBand(el, band) {
  el.textContent = band || "–";
  el.className = "hc-band " + (band || "").toLowerCase();
}

function renderLive(data) {
  const pill = $("status-pill");
  if (data.monitoring_active) {
    pill.className = "status-pill active";
    $("status-text").textContent = "● Monitoring Active";
    updateButtons(true);
  } else if (data.demo_runs > 0) {
    pill.className = "status-pill active";
    $("status-text").textContent = "● Demo Result Shown";
    updateButtons(false);
  } else {
    pill.className = "status-pill idle";
    $("status-text").textContent = "● Monitoring Stopped";
    updateButtons(false);
  }
  $("session-clock").textContent = fmtClock(data.elapsed_seconds || 0);

  // calibration banners
  const calibBanner = $("calibration-banner");
  const calibDone = $("calibration-done");
  if (data.calibrating && data.monitoring_active) {
    calibBanner.classList.remove("hidden");
    $("calibration-countdown").textContent =
      data.baseline && data.baseline.established
        ? "Collecting at least five valid windows per interaction mode."
        : Math.ceil(data.calibration_remaining / 60) + " min remaining.";
  } else {
    calibBanner.classList.add("hidden");
  }
  if (data.baseline && data.baseline.established &&
      data.behavioral_baseline && data.behavioral_baseline.established) {
    calibDone.classList.remove("hidden");
  } else {
    calibDone.classList.add("hidden");
  }

  // health cards
  const r = data.last_result;
  const collecting = !r;
  $("collecting-msg").classList.toggle("visible", collecting);
  if (collecting) {
    $("collecting-msg").textContent = data.insufficient
      ? "Insufficient activity in this window — move and type in the assessment to collect more data."
      : "Collecting more behavioural data…";
    ["v-fatigue", "v-stress", "v-anxiety", "v-motor"].forEach((id) => {
      $(id).textContent = "–";
    });
    ["b-fatigue", "b-stress", "b-anxiety", "b-motor"].forEach((id) => {
      setBand($(id), "INSUFFICIENT");
    });
    $("overall-line").classList.add("hidden");
    $("explanation").innerHTML = "";
    $("indicators").innerHTML = "";
  }
  if (r) {
    const isDemo = !!r.demo_scenario;
    const behavioural = r.behavioral_analysis || {};
    const behaviouralIndicators = behavioural.indicators || {};
    const cards = [
      ["fatigue", "v-fatigue", "b-fatigue", r.fatigue_score, r.fatigue_band],
      ["stress_tension", "v-stress", "b-stress", r.stress_score, r.stress_band],
      ["motor_consistency", "v-motor", "b-motor",
        r.motor_risk_score, r.motor_band],
    ];
    cards.forEach(([key, valueId, bandId, demoScore, demoBand]) => {
      const item = behaviouralIndicators[key];
      const score = isDemo ? demoScore : item && item.score;
      const status = isDemo ? demoBand : item ? item.status : "Insufficient data";
      $(valueId).textContent = score === null || score === undefined
        ? "–" : score + "%";
      setBand($(bandId), status);
    });
    $("v-anxiety").textContent = isDemo ? r.anxiety_score + "%" : "–";
    setBand($("b-anxiety"), isDemo ? r.anxiety_band : "Not scored");

    $("overall-line").classList.remove("hidden");
    const os = $("overall-status");
    const complete = cards.every(([key]) =>
      behaviouralIndicators[key] &&
      behaviouralIndicators[key].score !== null);
    const statuses = cards.map(([key]) =>
      behaviouralIndicators[key] && behaviouralIndicators[key].status);
    const behavioralStatus = complete
      ? (statuses.includes("Notable change") ? "Notable change"
        : statuses.includes("Mild change") ? "Mild change" : "Normal")
      : "Insufficient data";
    os.textContent = isDemo ? r.overall_status : behavioralStatus;
    os.className = isDemo ? r.overall_status : behavioralStatus;

    const ex = $("explanation");
    ex.innerHTML = "";
    const explanation = isDemo ? (r.explanation || []) : [
      "Indicators require keyboard and mouse agreement and are not medical diagnoses.",
    ];
    explanation.forEach((t) => {
      const li = document.createElement("li");
      li.textContent = t;
      ex.appendChild(li);
    });

    const rec = $("recommendation");
    rec.textContent = "💡 " + (isDemo
      ? (r.recommendation || "")
      : "Use the indicators as non-medical comparisons with your personal interaction-mode baseline.");
    rec.classList.remove("hidden");

    const ind = $("indicators");
    ind.innerHTML = "";
    Object.entries(behaviouralIndicators).forEach(([name, item]) => {
      const li = document.createElement("li");
      const score = item.score === null ? "—" : item.score + "%";
      li.textContent = name.replace(/_/g, " ") + ": " + item.status +
        " (" + score + ")";
      ind.appendChild(li);
    });
    (isDemo ? (r.indicators || []) : []).forEach((t) => {
      const li = document.createElement("li");
      li.textContent = "⚠ " + t;
      ind.appendChild(li);
    });
    if (behavioural.mode_status) {
      const li = document.createElement("li");
      li.textContent = "Interaction mode: " + behavioural.mode_status +
        (behavioural.mode ? " (" + behavioural.mode + ")" : "");
      ind.appendChild(li);
    }
  }

  // features
  const f = data.features;
  if (f) {
    $("m-wpm").textContent = fmtNum(f.typing_speed_wpm, 1);
    $("m-dwell").textContent = fmtNum(f.avg_dwell_time, 0);
    $("m-flight").textContent = fmtNum(f.avg_flight_time, 0);
    $("m-backspace").textContent = fmtNum(f.backspace_rate, 1);
    $("m-errors").textContent = fmtNum(f.error_rate, 1);
    $("m-tpause").textContent = fmtNum(f.typing_pause_duration, 0);
    $("m-mspeed").textContent = fmtNum(f.mouse_average_velocity, 0);
    $("m-accel").textContent = fmtNum(f.mouse_mean_absolute_acceleration, 0);
    $("m-jitter").textContent = fmtNum(f.mouse_jitter_index, 3);
    $("m-eff").textContent = fmtNum(f.mouse_path_efficiency, 2);
    $("m-clicks").textContent = fmtNum(f.click_rate, 1);
    $("m-acq").textContent = fmtNum(f.target_acquisition_time, 0);
  } else if (data.insufficient) {
    ["m-wpm", "m-dwell", "m-flight", "m-backspace", "m-errors", "m-tpause",
      "m-mspeed", "m-accel", "m-jitter", "m-eff", "m-clicks", "m-acq"]
      .forEach((id) => { $(id).textContent = "–"; });
  }

  // baseline panel
  const bl = data.baseline;
  if (bl) {
    const bs = $("baseline-status");
    bs.textContent = bl.established
      ? "✔ Established from " + bl.n_samples + " calibration window(s)"
      : (data.calibrating ? "Calibrating — complete the assessment to build it…" : "Not established yet");
    if (data.behavioral_baseline && !data.behavioral_baseline.established) {
      bs.textContent += " · Interaction-mode baseline provisional";
    }
    bs.className = "baseline-status" + (bl.established ? " established" : "");
    const d = bl.display || {};
    $("bl-wpm").textContent = fmtNum(d.typing_speed_wpm, 1);
    $("bl-dwell").textContent = fmtNum(d.avg_dwell_time, 0);
    $("bl-flight").textContent = fmtNum(d.avg_flight_time, 0);
    $("bl-errors").textContent = fmtNum(d.error_rate, 2);
    $("bl-mspeed").textContent = fmtNum(d.mouse_speed, 0);
    $("bl-jitter").textContent = fmtNum(d.mouse_jitter, 3);
    $("bl-acq").textContent = fmtNum(d.target_acquisition_time, 0);
    $("bl-samples").textContent = bl.n_samples;
  }

  if (data.trend) drawTrend(data.trend);
  renderReports(data.reports || []);

  const buf = (data.buffer) || {};
  $("live-events").textContent =
    (buf.key_events || 0) + " keyboard / " + (buf.mouse_points || 0) +
    " mouse events buffered · " + (buf.chars_typed || 0) + " chars counted · " +
    (buf.targets_hit || 0) + " targets hit (counters only — no content stored)";
}

// ---------------------------------------------------------------------------
// Reports
// ---------------------------------------------------------------------------

function renderReports(reports) {
  const empty = $("no-report");
  const content = $("report-content");
  const history = $("report-history");

  if (!reports.length) {
    empty.classList.add("visible");
    content.classList.add("hidden");
    return;
  }
  empty.classList.remove("visible");
  content.classList.remove("hidden");

  const newest = reports[0];
  $("report-range").textContent = newest.start_time + "  →  " + newest.end_time;
  const ro = $("report-overall");
  ro.textContent = newest.behavioral_status || "Insufficient data";
  ro.className = "report-overall " + ro.textContent.replace(/ /g, "-");
  const indicators = newest.behavioral_indicators || {};
  const reportScore = (key) => indicators[key] &&
    indicators[key].score !== null ? indicators[key].score : "–";
  $("r-fatigue").textContent = reportScore("fatigue");
  $("r-stress").textContent = reportScore("stress_tension");
  $("r-anxiety").textContent = "–";
  $("r-motor").textContent = reportScore("motor_consistency");

  const ch = $("report-changes");
  ch.innerHTML = "";
  (newest.behavioural_changes || []).forEach((t) => {
    const li = document.createElement("li");
    li.textContent = t;
    ch.appendChild(li);
  });
  Object.entries(newest.behavioral_indicators || {}).forEach(([name, item]) => {
    const li = document.createElement("li");
    li.textContent = name.replace(/_/g, " ") + ": " + item.status +
      " (" + (item.score === null ? "—" : item.score + "%") + ")";
    ch.appendChild(li);
  });
  $("report-recommendation").textContent =
    "Behavioral indicators are non-medical activity comparisons, not diagnoses.";

  history.innerHTML = "";
  reports.slice(0, 10).forEach((rep) => {
    const div = document.createElement("div");
    div.className = "report-item";
    const head = document.createElement("div");
    head.className = "ri-head";
    head.innerHTML = "<span>" + rep.start_time + "</span>" +
      "<b class='" + (rep.behavioral_status || "Insufficient data") + "'>" +
      (rep.behavioral_status || "Insufficient data") + "</b>";
    const scores = document.createElement("div");
    scores.className = "ri-scores";
    const repIndicators = rep.behavioral_indicators || {};
    const showReportScore = (key) => repIndicators[key] &&
      repIndicators[key].score !== null
        ? repIndicators[key].score + "%" : "–";
    scores.innerHTML =
      "<span class='ri-f'>F " + showReportScore("fatigue") + "</span>" +
      "<span class='ri-s'>S " + showReportScore("stress_tension") + "</span>" +
      "<span class='ri-a'>A –</span>" +
      "<span class='ri-m'>M " + showReportScore("motor_consistency") + "</span>";
    div.appendChild(head);
    div.appendChild(scores);
    history.appendChild(div);
  });
}

// ---------------------------------------------------------------------------
// SVG trend chart (no external libs)
// ---------------------------------------------------------------------------

function drawTrend(trend) {
  const svg = $("trend-chart");
  const W = 640, H = 260, PAD_L = 34, PAD_R = 8, PAD_T = 12, PAD_B = 24;
  const labels = trend.labels || [];
  const series = [
    { key: "fatigue", color: "var(--fatigue)", vals: trend.fatigue || [] },
    { key: "stress/tension", color: "var(--stress)", vals: trend.stress || [] },
    { key: "anxiety (not scored)", color: "var(--anxiety)", vals: trend.anxiety || [] },
    { key: "motor consistency", color: "var(--motor)", vals: trend.motor || [] },
  ];
  const n = Math.max(labels.length, 1);
  const x = (i) => PAD_L + (i / Math.max(n - 1, 1)) * (W - PAD_L - PAD_R);
  const y = (v) => PAD_T + (1 - v / 100) * (H - PAD_T - PAD_B);

  let g = "";
  for (let gv = 0; gv <= 100; gv += 25) {
    g += "<line x1='" + PAD_L + "' y1='" + y(gv) + "' x2='" + (W - PAD_R) +
      "' y2='" + y(gv) + "' stroke='rgba(29,43,37,.8)' stroke-width='1'/>";
    g += "<text x='" + (PAD_L - 6) + "' y='" + (y(gv) + 4) +
      "' fill='#91A39A' font-size='10' text-anchor='end'>" + gv + "</text>";
  }
  g += "<line x1='" + PAD_L + "' y1='" + y(30) + "' x2='" + (W - PAD_R) +
    "' y2='" + y(30) + "' stroke='rgba(110,220,138,.28)' stroke-dasharray='4 4'/>";
  g += "<line x1='" + PAD_L + "' y1='" + y(60) + "' x2='" + (W - PAD_R) +
    "' y2='" + y(60) + "' stroke='rgba(229,115,115,.28)' stroke-dasharray='4 4'/>";

  series.forEach((s) => {
    if (!s.vals.length) return;
    let d = "";
    let penDown = false;
    s.vals.forEach((v, i) => {
      if (v === null || v === undefined || !Number.isFinite(Number(v))) {
        penDown = false;
        return;
      }
      d += (penDown ? "L" : "M") + x(i).toFixed(1) + "," + y(v).toFixed(1);
      penDown = true;
    });
    if (d) {
      g += "<path d='" + d + "' fill='none' stroke='" + s.color +
        "' stroke-width='2' stroke-linejoin='round' stroke-linecap='round'/>";
      for (let i = s.vals.length - 1; i >= 0; i--) {
        if (s.vals[i] !== null && s.vals[i] !== undefined &&
            Number.isFinite(Number(s.vals[i]))) {
          g += "<circle cx='" + x(i).toFixed(1) + "' cy='" + y(s.vals[i]).toFixed(1) +
            "' r='3.5' fill='" + s.color + "'/>";
          break;
        }
      }
    }
  });

  if (labels.length) {
    const step = Math.ceil(labels.length / 8);
    labels.forEach((lb, i) => {
      if (i % step === 0 || i === labels.length - 1) {
        g += "<text x='" + x(i).toFixed(1) + "' y='" + (H - 6) +
          "' fill='#91A39A' font-size='10' text-anchor='middle'>" + lb + "</text>";
      }
    });
  }

  svg.innerHTML = g;
}

// ---------------------------------------------------------------------------
// Polling
// ---------------------------------------------------------------------------

async function poll() {
  try {
    const res = await fetch(API.live);
    if (!res.ok) return;
    renderLive(await res.json());
  } catch (err) { /* backend briefly unavailable */ }
}

// ---------------------------------------------------------------------------
// Wire up
// ---------------------------------------------------------------------------

$("btn-start").addEventListener("click", startMonitoring);
$("btn-stop").addEventListener("click", stopMonitoring);
$("btn-reset").addEventListener("click", resetSession);
$("btn-open-assessment").addEventListener("click", () => {
  window.open(API.assessment, "_blank", "noopener");
});
document.querySelectorAll(".btn.demo").forEach((b) =>
  b.addEventListener("click", () => runDemo(b.dataset.scenario)));

state.pollTimer = setInterval(poll, 2000);
poll();
