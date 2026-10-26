/*
 * AI Health Monitor - Behavioural Health Assessment (frontend)
 * ============================================================
 * PRIVACY (hard requirement):
 *   - The typed sentence and its characters exist ONLY in this page's DOM.
 *     We never transmit .value, key values, or any text.
 *   - What leaves the browser: key-down/key-up TIMESTAMPS (ms), a backspace
 *     flag, aggregate counters (chars typed, errors), mouse coordinates
 *     inside the designated area, timestamps, and target-acquisition DURATIONS.
 *
 * The behavioural analysis itself runs server-side through the SAME pipeline
 * used everywhere else: feature extraction -> baseline -> ML model -> fusion
 * -> report engine. This page only captures events and shows progress.
 */

"use strict";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const API = {
  events: "/api/events",
  live: "/api/live",
  status: "/api/status",
  start: "/api/start",
  stop: "/api/stop",
  sentences: "/api/sentences",
};

const TOTAL_SENTENCES = 10;

const state = {
  sentences: [],
  index: 0,              // current sentence idx
  completed: [],         // per-sentence local stats
  sentenceStart: null,   // performance.now() at sentence start
  assessStart: performance.now(),
  totalErrors: 0,
  totalChars: 0,
  sentChars: 0,          // chars counter already shipped
  sentErrors: 0,
  keyEvents: [],         // {t_down, t_up, is_backspace} ms (send buffer)
  sentenceEvents: [],    // same event objects, kept per-sentence for stats
  mouseEvents: [],       // {x, y, t, clicked} ms
  targetAcquisitions: [],// durations ms
  downMap: new Map(),
  lastKeyUp: null,
  // rolling local display stats
  dwellSum: 0, dwellN: 0, flightSum: 0, flightN: 0,
  keyTimes: [],          // recent key-down times for local WPM
  mouseSpeeds: [],       // recent px/s samples
  lastMouse: null,
  running: false,
  finished: false,
  continueMode: false,
  targetTimer: null,
  activeTargets: 0,
  targetsHit: 0,
};

const $ = (id) => document.getElementById(id);

const BACKSPACE_CODES = new Set([8, 46]);
const IGNORED_CODES = new Set([16, 17, 18, 20, 27, 91, 92, 93, 144, 145, 9]);

// ---------------------------------------------------------------------------
// Keyboard timing capture (privacy-safe: timing only)
// ---------------------------------------------------------------------------

function onKeyDown(e) {
  if (!state.running) return;
  if (IGNORED_CODES.has(e.keyCode)) return;
  const now = performance.now();
  state.keyTimes.push(now);
  let ev;
  if (BACKSPACE_CODES.has(e.keyCode) && !state.downMap.has(e.keyCode)) {
    ev = { t_down: now, t_up: now, is_backspace: true };
  } else if (!state.downMap.has(e.keyCode)) {
    state.downMap.set(e.keyCode, now);
    ev = { t_down: now, t_up: null, is_backspace: false };
  } else {
    return;
  }
  // the SAME object goes into both arrays: shipping clears the send buffer
  // but per-sentence stats keep their references
  state.keyEvents.push(ev);
  state.sentenceEvents.push(ev);
}

function onKeyUp(e) {
  if (!state.running) return;
  const t0 = state.downMap.get(e.keyCode);
  if (t0 === undefined) return;
  const tUp = performance.now();
  for (let i = state.keyEvents.length - 1; i >= 0; i--) {
    const ev = state.keyEvents[i];
    if (ev.t_up === null && Math.abs(ev.t_down - t0) < 1) {
      ev.t_up = tUp;
      const dwell = ev.t_up - ev.t_down;
      if (dwell >= 10 && dwell <= 3000) { state.dwellSum += dwell; state.dwellN += 1; }
      if (state.lastKeyUp !== null && !ev.is_backspace) {
        const flight = ev.t_down - state.lastKeyUp;
        if (flight >= 5 && flight <= 5000) { state.flightSum += flight; state.flightN += 1; }
      }
      if (!ev.is_backspace) state.lastKeyUp = tUp;
      break;
    }
  }
  state.downMap.delete(e.keyCode);
}

document.addEventListener("keydown", onKeyDown, { passive: true });
document.addEventListener("keyup", onKeyUp, { passive: true });

// ---------------------------------------------------------------------------
// Typing test
// ---------------------------------------------------------------------------

function renderSentence() {
  const target = state.sentences[state.index] || "";
  const box = $("sentence-display");
  box.innerHTML = "";
  for (const ch of target) {
    const span = document.createElement("span");
    span.className = "char char-pending";
    span.textContent = ch;
    if (ch === " ") span.classList.add("char-space");
    box.appendChild(span);
  }
  $("sentence-progress").textContent =
    "Sentence " + (state.index + 1) + " / " + TOTAL_SENTENCES;
  $("progress-fill").style.width = (state.index / TOTAL_SENTENCES * 100) + "%";
  $("typing-input").value = "";
  $("typing-feedback").textContent =
    "Type the sentence exactly as shown. Corrections are allowed.";
  state.sentenceStart = performance.now();
  state.sentenceErrors = 0;
  state.sentenceEvents = [];
  state.lastKeyUp = null;
}

function onTypingInput() {
  if (!state.running || state.finished) return;
  const input = $("typing-input");
  const target = state.sentences[state.index] || "";
  const typed = input.value;
  const spans = $("sentence-display").children;
  let allCorrect = true;

  for (let i = 0; i < spans.length; i++) {
    const span = spans[i];
    span.classList.remove("char-correct", "char-wrong", "char-cursor");
    if (i < typed.length) {
      if (typed[i] === target[i]) span.classList.add("char-correct");
      else { span.classList.add("char-wrong"); allCorrect = false; }
    } else {
      span.classList.add("char-pending");
      if (i === typed.length) span.classList.add("char-cursor");
      allCorrect = false;
    }
  }

  // error + char counters: compare with previous length (aggregate only)
  const prevLen = state._prevLen || 0;
  if (typed.length > prevLen) {
    const added = typed.length - prevLen;
    state.totalChars += added;
    let wrong = 0;
    for (let i = Math.max(prevLen, typed.length - added); i < typed.length; i++) {
      if (typed[i] !== target[i]) wrong++;
    }
    if (wrong > 0) {
      state.totalErrors += wrong;
      state.sentenceErrors += wrong;
      $("typing-feedback").textContent =
        wrong + (wrong === 1 ? " character" : " characters") +
        " off so far — keep going, you can fix anything with Backspace.";
    } else {
      $("typing-feedback").textContent = "Looking good — keep the rhythm.";
    }
  }
  state._prevLen = typed.length;

  // live WPM (local display only)
  const wpm = localWpm();
  $("wpm-live").textContent = (wpm > 0 ? Math.round(wpm) : "–") + " WPM";

  // sentence complete: full length AND every char correct
  if (typed.length === target.length && allCorrect) completeSentence();
}

function localWpm() {
  const now = performance.now();
  state.keyTimes = state.keyTimes.filter((t) => now - t < 15000);
  if (state.keyTimes.length < 4) return 0;
  const spanMin = Math.max((now - state.keyTimes[0]) / 60000, 0.02);
  return (state.keyTimes.length / 5) / spanMin;
}

function completeSentence() {
  const now = performance.now();
  const durS = state.sentenceStart ? (now - state.sentenceStart) / 1000 : 0;
  const keys = state.sentenceEvents.filter(
    (k) => !k.is_backspace && k.t_down >= (state.sentenceStart || 0));
  const nKeys = Math.max(keys.length, 1);
  const wpm = (nKeys / 5) / Math.max(durS / 60, 1 / 60);
  const dwells = keys.filter((k) => k.t_up !== null).map((k) => k.t_up - k.t_down);
  const avgDwell = dwells.length ? dwells.reduce((a, b) => a + b, 0) / dwells.length : 0;
  const flights = [];
  for (let i = 1; i < keys.length; i++) {
    if (keys[i - 1].t_up !== null) flights.push(keys[i].t_down - keys[i - 1].t_up);
  }
  const avgFlight = flights.length ? flights.reduce((a, b) => a + b, 0) / flights.length : 0;

  const stat = {
    n: state.index + 1,
    wpm: Math.round(wpm),
    dwell: Math.round(avgDwell),
    flight: Math.round(avgFlight),
    errors: state.sentenceErrors || 0,
    seconds: Math.round(durS * 10) / 10,
  };
  state.completed.push(stat);
  addSentenceLog(stat);

  state.index += 1;
  state._prevLen = 0;
  $("progress-fill").style.width = (state.index / TOTAL_SENTENCES * 100) + "%";

  if (state.index >= TOTAL_SENTENCES || state.index >= state.sentences.length) {
    finishAssessment();
  } else {
    renderSentence();
    $("typing-input").focus();
  }
}

function addSentenceLog(s) {
  const log = $("sentence-log");
  const empty = log.querySelector(".collecting");
  if (empty) empty.remove();
  const div = document.createElement("div");
  div.className = "sl-item";
  const head = document.createElement("div");
  head.className = "sl-head";
  head.textContent = "Sentence " + s.n + " completed";
  const stats = document.createElement("div");
  stats.className = "sl-stats";
  stats.innerHTML =
    "<span>Speed: <b>" + s.wpm + " WPM</b></span>" +
    "<span>Dwell: <b>" + s.dwell + " ms</b></span>" +
    "<span>Flight: <b>" + s.flight + " ms</b></span>" +
    "<span>Errors: <b>" + s.errors + "</b></span>" +
    "<span>Time: <b>" + s.seconds + " s</b></span>";
  div.appendChild(head);
  div.appendChild(stats);
  log.prepend(div);
}

// ---------------------------------------------------------------------------
// Mouse area capture + target task
// ---------------------------------------------------------------------------

function areaHandler(areaId) {
  const area = $(areaId);
  if (!area) return null;
  return {
    area,
    onMove(e) {
      if (!state.running) return;
      const rect = area.getBoundingClientRect();
      if (e.clientX < rect.left || e.clientX > rect.right ||
          e.clientY < rect.top || e.clientY > rect.bottom) return;
      const x = Math.round(e.clientX - rect.left);
      const y = Math.round(e.clientY - rect.top);
      const t = performance.now();
      // rolling speed for local display
      if (state.lastMouse) {
        const dt = t - state.lastMouse.t;
        if (dt > 5) {
          const sp = Math.hypot(x - state.lastMouse.x, y - state.lastMouse.y) / dt * 1000;
          if (sp < 20000) {
            state.mouseSpeeds.push(sp);
            if (state.mouseSpeeds.length > 300) state.mouseSpeeds.shift();
          }
        }
      }
      state.lastMouse = { x, y, t };
      state.mouseEvents.push({ x, y, t, clicked: false });
      if (state.mouseEvents.length > 4000) state.mouseEvents.splice(0, 1000);
    },
    onClick(e) {
      if (!state.running) return;
      const rect = area.getBoundingClientRect();
      if (e.clientX < rect.left || e.clientX > rect.right ||
          e.clientY < rect.top || e.clientY > rect.bottom) return;
      const x = Math.round(e.clientX - rect.left);
      const y = Math.round(e.clientY - rect.top);
      const t = performance.now();
      const evs = state.mouseEvents;
      if (evs.length && Math.abs(evs[evs.length - 1].t - t) < 150) {
        evs[evs.length - 1].clicked = true;
      } else {
        evs.push({ x, y, t, clicked: true });
      }
    },
  };
}

const mainMouse = areaHandler("mouse-area");
const freeMouse = areaHandler("mouse-area-free");
[mainMouse, freeMouse].forEach((h) => {
  if (!h) return;
  h.area.addEventListener("mousemove", h.onMove);
  h.area.addEventListener("click", h.onClick);
});

function spawnTarget() {
  if (!state.running || state.finished || state.continueMode) return;
  const layer = $("target-layer");
  const area = $("mouse-area");
  const w = area.clientWidth, h = area.clientHeight;
  if (state.activeTargets >= 2) return;
  const el = document.createElement("div");
  el.className = "target" + (Math.random() < 0.4 ? " shrink" : "");
  const x = 30 + Math.random() * Math.max(w - 60, 10);
  const y = 30 + Math.random() * Math.max(h - 60, 10);
  el.style.left = x + "px";
  el.style.top = y + "px";
  const spawnAt = performance.now();
  el.addEventListener("click", () => {
    if (el._hit) return;
    el._hit = true;
    el.classList.add("hit");
    state.targetsHit += 1;
    state.activeTargets -= 1;
    state.targetAcquisitions.push(performance.now() - spawnAt);
    $("targets-hit").textContent = state.targetsHit;
    $("targets-active").textContent = state.activeTargets;
    const acqs = state.targetAcquisitions;
    $("avg-acq").textContent = Math.round(
      acqs.reduce((a, b) => a + b, 0) / acqs.length);
    setTimeout(() => el.remove(), 480);
    $("mouse-hint").textContent = "Nice — next target incoming…";
    setTimeout(spawnTarget, 700);
  });
  layer.appendChild(el);
  state.activeTargets += 1;
  $("targets-active").textContent = state.activeTargets;
  $("mouse-hint").textContent = "Click the highlighted target!";
  // stale target: remove after 8 s and spawn a fresh one
  setTimeout(() => {
    if (!el._hit && el.isConnected) {
      el.remove();
      state.activeTargets -= 1;
      $("targets-active").textContent = state.activeTargets;
      spawnTarget();
    }
  }, 8000);
}

// ---------------------------------------------------------------------------
// Event shipping (batch every 2.5 s) — timing + aggregate counts ONLY
// ---------------------------------------------------------------------------

function sendEvents() {
  if (!state.running) return;
  const keys = state.keyEvents;
  const mouse = state.mouseEvents;
  const acq = state.targetAcquisitions;
  const charsDelta = state.totalChars - state.sentChars;
  const errDelta = state.totalErrors - state.sentErrors;
  if (keys.length === 0 && mouse.length === 0 && acq.length === 0 &&
      charsDelta === 0 && errDelta === 0) return;
  state.keyEvents = [];
  state.mouseEvents = [];
  state.targetAcquisitions = [];
  state.sentChars = state.totalChars;
  state.sentErrors = state.totalErrors;
  fetch(API.events, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      keyboard: keys.map((k) => ({ t_down: k.t_down, t_up: k.t_up, is_backspace: k.is_backspace })),
      mouse,
      n_chars_typed: charsDelta,
      n_errors: errDelta,
      target_acquisitions: acq,
    }),
  }).catch(() => {});
}

// ---------------------------------------------------------------------------
// Polling + rendering
// ---------------------------------------------------------------------------

function fmtClock(sec) {
  const m = Math.floor(sec / 60), s = sec % 60;
  return String(m).padStart(2, "0") + ":" + String(s).padStart(2, "0");
}

function fmtNum(v, d) {
  if (v === null || v === undefined || isNaN(v)) return "–";
  return Number(v).toFixed(d);
}

function localMetrics() {
  const avg = (s, n) => (n ? s / n : null);
  const msSpeeds = state.mouseSpeeds.slice(-200);
  return {
    wpm: localWpm(),
    dwell: avg(state.dwellSum, state.dwellN),
    flight: avg(state.flightSum, state.flightN),
    mspeed: msSpeeds.length
      ? msSpeeds.reduce((a, b) => a + b, 0) / msSpeeds.length : null,
    jitter: null,
    acq: state.targetAcquisitions.length
      ? state.targetAcquisitions.reduce((a, b) => a + b, 0) / state.targetAcquisitions.length
      : null,
  };
}

function renderLive(data) {
  const pill = $("status-pill");
  if (data.monitoring_active) {
    pill.className = "status-pill active";
    $("status-text").textContent = "● Monitoring Active";
  } else {
    pill.className = "status-pill idle";
    $("status-text").textContent = "● Monitoring Stopped";
  }
  $("session-clock").textContent = fmtClock(data.elapsed_seconds || 0);
  $("assess-timer").textContent = fmtClock(data.elapsed_seconds || 0);

  // calibration banners
  const est = data.baseline && data.baseline.established;
  $("calib-banner").classList.toggle("hidden", !(data.calibrating && data.monitoring_active));
  $("calib-done").classList.toggle("hidden", !est);
  $("calib-none").classList.toggle("hidden", est || data.calibrating);
  if (data.calibrating && data.monitoring_active) {
    $("calib-countdown").textContent =
      Math.ceil((data.calibration_remaining || 0) / 60) + " min remaining.";
  }

  // metrics: prefer server features (analysis truth), fill gaps with local
  const f = data.features || {};
  const local = localMetrics();
  const pick = (serverV, localV, d) =>
    (serverV !== undefined && serverV !== null ? fmtNum(serverV, d)
      : (localV !== null && localV !== undefined ? fmtNum(localV, d) : "–"));
  const ids = state.continueMode
    ? { wpm: "f-wpm", dwell: "f-dwell", flight: "f-flight", mspeed: "f-mspeed" }
    : { wpm: "m-wpm", dwell: "m-dwell", flight: "m-flight", mspeed: "m-mspeed" };
  $(ids.wpm).textContent = pick(f.typing_speed_wpm, local.wpm, 1);
  $(ids.dwell).textContent = pick(f.avg_dwell_time, local.dwell, 0);
  $(ids.flight).textContent = pick(f.avg_flight_time, local.flight, 0);
  $(ids.mspeed).textContent = pick(f.mouse_speed, local.mspeed, 0);
  if (!state.continueMode) {
    $("m-backspace").textContent = fmtNum(f.backspace_rate, 1);
    $("m-jitter").textContent = fmtNum(f.mouse_jitter, 3);
    $("m-acq").textContent = fmtNum(
      f.target_acquisition_time !== undefined ? f.target_acquisition_time : local.acq, 0);
    $("m-errors").textContent = state.totalErrors;
  }

  // current behavioural state
  const r = data.last_result;
  const hasResult = !!r;
  $("collecting-msg").classList.toggle("visible", !hasResult);
  $("assess-current").classList.toggle("hidden", !hasResult);
  if (hasResult) {
    $("ac-fatigue").textContent = r.fatigue_score + "%";
    $("ac-stress").textContent = r.stress_score + "%";
    $("ac-anxiety").textContent = r.anxiety_score + "%";
    $("ac-motor").textContent = r.motor_risk_score + "%";
    $("ac-overall").textContent = r.overall_status;
  }
}

async function poll() {
  try {
    const res = await fetch(API.live);
    if (!res.ok) return;
    renderLive(await res.json());
  } catch (err) { /* backend briefly unavailable */ }
}

// ---------------------------------------------------------------------------
// Completion & views
// ---------------------------------------------------------------------------

function finishAssessment() {
  state.finished = true;
  clearInterval(state.targetTimer);
  $("target-layer").innerHTML = "";
  state.activeTargets = 0;

  const c = state.completed;
  const totalS = c.reduce((a, s) => a + s.seconds, 0);
  const avg = (fn) => (c.length ? c.reduce((a, s) => a + fn(s), 0) / c.length : 0);
  const errRate = state.totalChars ? state.totalErrors / state.totalChars * 100 : 0;
  const ms = localMetrics();

  $("complete-progress").textContent = c.length + " / " + c.length + " sentences completed";
  $("c-time").textContent = Math.round(totalS);
  $("c-wpm").textContent = Math.round(avg((s) => s.wpm));
  $("c-dwell").textContent = Math.round(avg((s) => s.dwell));
  $("c-flight").textContent = Math.round(avg((s) => s.flight));
  $("c-errors").textContent = errRate.toFixed(1);
  $("c-targets").textContent = state.targetsHit;
  $("c-mspeed").textContent = ms.mspeed !== null ? Math.round(ms.mspeed) : "–";
  $("c-jitter").textContent = "see dashboard";

  $("view-assessment").classList.add("hidden");
  $("view-complete").classList.remove("hidden");
  // keep monitoring running server-side; keep shipping buffered events
  sendEvents();
}

function showContinue() {
  state.continueMode = true;
  $("view-complete").classList.add("hidden");
  $("view-continue").classList.remove("hidden");
  $("free-typing").focus();
}

async function returnToDashboard() {
  // Stop the assessment capture locally; server session keeps running so the
  // dashboard keeps its live analysis (per spec: do not stop the app).
  state.running = false;
  clearInterval(state.targetTimer);
  sendEvents();
  try { window.close(); } catch (e) { /* popup blocker */ }
  // fallback when the tab was not opener-created
  setTimeout(() => { window.location.href = "/"; }, 150);
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

async function boot() {
  // make sure the server session is active (user already clicked Start)
  try {
    const st = await fetch(API.status).then((r) => r.json());
    if (!st.monitoring_active) await fetch(API.start, { method: "POST" });
    state.running = true;
  } catch (err) {
    $("status-text").textContent = "● Backend unavailable";
    return;
  }

  // fetch the random 10 sentences
  try {
    const data = await fetch(API.sentences + "?n=" + TOTAL_SENTENCES).then((r) => r.json());
    state.sentences = data.sentences || [];
  } catch (err) {
    state.sentences = [];
  }
  if (!state.sentences.length) {
    $("sentence-display").textContent = "Could not load sentences — is the backend running?";
    return;
  }

  renderSentence();
  $("typing-input").addEventListener("input", onTypingInput);
  $("typing-input").focus();

  $("btn-return").addEventListener("click", returnToDashboard);
  $("btn-continue").addEventListener("click", showContinue);
  $("btn-dashboard").addEventListener("click", () => { window.location.href = "/"; });
  $("btn-stop-assessment").addEventListener("click", async () => {
    state.running = false;
    clearInterval(state.targetTimer);
    try { await fetch(API.stop, { method: "POST" }); } catch (e) {}
    $("status-pill").className = "status-pill idle";
    $("status-text").textContent = "● Monitoring Stopped";
  });

  // target task: first target shortly after arrival, then chain on hit
  setTimeout(spawnTarget, 1200);
  // safety spawner so the area never sits empty
  state.targetTimer = setInterval(() => {
    if (state.activeTargets === 0) spawnTarget();
  }, 5000);

  setInterval(sendEvents, 2500);
  setInterval(poll, 2000);
  poll();
}

boot();
