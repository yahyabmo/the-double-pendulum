import "./style.css";
import { Simulation, bobPositions, totalEnergy, type PendulumState } from "./physics/sim";
import { Renderer, type BobId, type TrailMode } from "./render/renderer";
import { PLANETS, PRESETS, type Preset } from "./presets";
import { chips, h, section, segmented, slider, toggle, type SliderHandle } from "./ui/controls";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const canvas = document.getElementById("stage") as HTMLCanvasElement;
const panel = document.getElementById("panel")!;
const hud = document.getElementById("hud")!;
const hint = document.getElementById("hint")!;
const renderer = new Renderer(canvas);

const first = PRESETS[0];
const sim = new Simulation(first.params);

const ui = {
  playing: true,
  speed: 1,
  rewinding: false,
  trailSeconds: first.trailSeconds ?? 4,
  trailMode: "speed" as TrailMode,
  innerTrail: false,
  showArms: true,
  hover: null as BobId | null,
  activePreset: first.id as string | null,
  panelOpen: window.innerWidth > 760,
};

interface DragState {
  bob: BobId;
  pointerId: number;
  /** Continuous (unwrapped) angles sampled while dragging, for fling velocity. */
  samples: { t: number; th1: number; th2: number }[];
  th1: number;
  th2: number;
}
let drag: DragState | null = null;

const deg = 180 / Math.PI;
const wrapPi = (a: number) => a - 2 * Math.PI * Math.floor((a + Math.PI) / (2 * Math.PI));

// ---------------------------------------------------------------------------
// Panel
// ---------------------------------------------------------------------------
const sliders: Record<string, SliderHandle> = {};
let presetButtons: HTMLButtonElement[] = [];
let planetChips: { el: HTMLElement; refresh(): void };

function markCustom(): void {
  ui.activePreset = null;
  presetButtons.forEach((b) => b.classList.remove("active"));
}

function buildPanel(): void {
  // Presets
  const pre = section("Presets");
  const grid = h("div", { class: "presets" });
  presetButtons = PRESETS.map((p, i) => {
    const b = h("button", { type: "button", class: "preset" }, h("kbd", {}, String(i + 1)), h("b", {}, p.name), h("span", {}, p.blurb));
    b.addEventListener("click", () => applyPreset(p));
    grid.append(b);
    return b;
  });
  pre.body.append(grid);

  // Physics
  const phys = section("Physics", true, "Changes apply live — the motion carries on with the new physics.");
  const live = (key: keyof typeof sim.params) => (v: number) => {
    sim.params[key] = v;
    sim.rebaseEnergy();
    markCustom();
    planetChips.refresh();
  };
  sliders.g = slider({ label: "Gravity  g", min: 0, max: 30, value: sim.params.g, format: (v) => `${v.toFixed(2)} m/s²`, onInput: live("g") });
  planetChips = chips(
    PLANETS.map((p) => ({ value: p.g, label: p.label, title: `g = ${p.g} m/s²` })),
    (g) => sliders.g.set(g, false),
    (g) => Math.abs(sim.params.g - g) < 1e-6,
  );
  sliders.m1 = slider({ label: "Mass 1", min: 0.1, max: 10, log: true, value: sim.params.m1, format: (v) => `${v.toFixed(2)} kg`, onInput: live("m1") });
  sliders.m2 = slider({ label: "Mass 2", min: 0.1, max: 10, log: true, value: sim.params.m2, format: (v) => `${v.toFixed(2)} kg`, onInput: live("m2") });
  sliders.L1 = slider({ label: "Length 1", min: 0.2, max: 3, value: sim.params.L1, format: (v) => `${v.toFixed(2)} m`, onInput: live("L1") });
  sliders.L2 = slider({ label: "Length 2", min: 0.2, max: 3, value: sim.params.L2, format: (v) => `${v.toFixed(2)} m`, onInput: live("L2") });
  sliders.b = slider({ label: "Joint damping", min: 0, max: 1, value: sim.params.b, format: (v) => (v === 0 ? "none" : v.toFixed(3)), onInput: live("b") });
  phys.body.append(sliders.g.el, planetChips.el, sliders.m1.el, sliders.m2.el, sliders.L1.el, sliders.L2.el, sliders.b.el);

  // Starting position
  const ic = section("Starting position", false, "Restarts from these angles — or just drag the bobs on the canvas.");
  const fromIC = () => {
    sim.reseed({ th1: sliders.th1.get() / deg, w1: sliders.w1.get(), th2: sliders.th2.get() / deg, w2: sliders.w2.get() });
    markCustom();
  };
  const angleFmt = (v: number) => `${v.toFixed(1)}°`;
  sliders.th1 = slider({ label: "θ₁", min: -180, max: 180, value: 0, format: angleFmt, onInput: fromIC });
  sliders.th2 = slider({ label: "θ₂", min: -180, max: 180, value: 0, format: angleFmt, onInput: fromIC });
  sliders.w1 = slider({ label: "ω₁", min: -15, max: 15, value: 0, format: (v) => `${v.toFixed(2)} rad/s`, onInput: fromIC });
  sliders.w2 = slider({ label: "ω₂", min: -15, max: 15, value: 0, format: (v) => `${v.toFixed(2)} rad/s`, onInput: fromIC });
  ic.body.append(sliders.th1.el, sliders.th2.el, sliders.w1.el, sliders.w2.el);

  // Butterfly
  const bf = section(
    "Butterfly effect",
    true,
    "Run many pendulums at once, each nudged by a tiny Δθ₂. They move as one… until they don't.",
  );
  const restartEnsemble = () => {
    const sign = Math.sign(sim.spread) || 1;
    sim.reseed(sim.seed, Math.round(sliders.count.get()), sign * sliders.spread.get());
    markCustom();
  };
  sliders.count = slider({ label: "Pendulums", min: 1, max: 2000, log: true, step: 1, value: 1, format: (v) => String(Math.round(v)), onInput: restartEnsemble });
  sliders.spread = slider({ label: "Δθ₂ between them", min: 1e-12, max: 1e-1, log: true, value: 1e-6, format: (v) => `${v.toExponential(0)} rad`, onInput: restartEnsemble });
  const fork = h("button", { type: "button", class: "btn", title: "Split the current pendulum into an ensemble right now" }, "Fork from current motion");
  fork.addEventListener("click", () => {
    const n = sliders.count.get() < 2 ? 100 : Math.round(sliders.count.get());
    sim.reseed(sim.pendulum0, n, sliders.spread.get());
    syncPanel();
    markCustom();
  });
  bf.body.append(sliders.count.el, h("div", { class: "spectrum", title: "Ensemble colours: first → last" }), sliders.spread.el, fork);

  // Display
  const disp = section("Display");
  sliders.trail = slider({ label: "Trail length", min: 0, max: 12, value: ui.trailSeconds, format: (v) => (v === 0 ? "off" : `${v.toFixed(1)} s`), onInput: (v) => (ui.trailSeconds = v) });
  const modeHost = h("div", {});
  segmented<TrailMode>(
    modeHost,
    [
      { value: "speed", label: "Speed" },
      { value: "rainbow", label: "Rainbow" },
      { value: "solid", label: "Solid" },
    ],
    ui.trailMode,
    (m) => (ui.trailMode = m),
  );
  const modeRow = h("div", { class: "row" }, h("label", {}, "Trail colour"), modeHost);
  disp.body.append(
    sliders.trail.el,
    modeRow,
    toggle("Trail on inner bob", ui.innerTrail, (v) => (ui.innerTrail = v)).el,
    toggle("Show arms", ui.showArms, (v) => (ui.showArms = v)).el,
  );

  // Accuracy
  const acc = section("Accuracy", false, "Fixed-step RK4 integrator. More steps → less energy drift, more CPU.");
  sliders.sps = slider({
    label: "Steps per second",
    min: 60,
    max: 6000,
    log: true,
    step: 1,
    value: sim.opts.stepsPerSecond,
    format: (v) => String(Math.round(v)),
    onInput: (v) => {
      sim.setStepsPerSecond(Math.round(v));
      sim.rebaseEnergy();
    },
  });
  acc.body.append(sliders.sps.el);

  // About
  const about = section("About", false);
  about.body.append(
    h(
      "p",
      { class: "section-note" },
      "The equations of motion are derived symbolically from the Lagrangian L = T − V with SymPy (physics/derive.py), " +
        "then compiled to TypeScript. Damping enters as a generalized force at each joint.",
    ),
  );

  panel.append(pre.el, phys.el, ic.el, bf.el, disp.el, acc.el, about.el);
}

/** Push simulation values back into the controls (after presets / drags). */
function syncPanel(): void {
  const p = sim.params;
  sliders.g.set(p.g);
  sliders.m1.set(p.m1);
  sliders.m2.set(p.m2);
  sliders.L1.set(p.L1);
  sliders.L2.set(p.L2);
  sliders.b.set(p.b);
  sliders.th1.set(wrapPi(sim.seed.th1) * deg);
  sliders.th2.set(wrapPi(sim.seed.th2) * deg);
  sliders.w1.set(sim.seed.w1);
  sliders.w2.set(sim.seed.w2);
  sliders.count.set(sim.n);
  sliders.spread.set(Math.abs(sim.spread) || 1e-6);
  sliders.trail.set(ui.trailSeconds);
  planetChips.refresh();
  presetButtons.forEach((b, i) => b.classList.toggle("active", PRESETS[i].id === ui.activePreset));
}

function applyPreset(p: Preset): void {
  sim.params = { ...p.params };
  sim.reseed(p.state, p.count, p.spread);
  if (p.trailSeconds !== undefined) ui.trailSeconds = p.trailSeconds;
  ui.activePreset = p.id;
  ui.playing = true;
  syncPlay();
  syncPanel();
}

// ---------------------------------------------------------------------------
// Transport
// ---------------------------------------------------------------------------
const playBtn = document.getElementById("btn-play")!;
const rewindBtn = document.getElementById("btn-rewind")!;

function syncPlay(): void {
  document.body.classList.toggle("paused", !ui.playing);
  playBtn.setAttribute("aria-pressed", String(!ui.playing));
}
function togglePlay(): void {
  ui.playing = !ui.playing;
  syncPlay();
}
function stepForward(): void {
  ui.playing = false;
  syncPlay();
  sim.advance(1 / 60, 1e9);
}
function setRewinding(v: boolean): void {
  ui.rewinding = v;
  rewindBtn.classList.toggle("held", v);
}
function clearTrails(): void {
  sim.history.clear();
  sim.history.push(sim.state, sim.t);
}
function setPanel(open: boolean): void {
  ui.panelOpen = open;
  document.body.classList.toggle("panel-closed", !open);
  layout();
}

playBtn.addEventListener("click", togglePlay);
document.getElementById("btn-reset")!.addEventListener("click", () => sim.reset());
document.getElementById("btn-step")!.addEventListener("click", stepForward);
document.getElementById("btn-panel")!.addEventListener("click", () => setPanel(!ui.panelOpen));
rewindBtn.addEventListener("pointerdown", (e) => {
  rewindBtn.setPointerCapture(e.pointerId);
  setRewinding(true);
});
for (const ev of ["pointerup", "pointercancel", "lostpointercapture"]) rewindBtn.addEventListener(ev, () => setRewinding(false));

segmented(
  document.getElementById("speed")!,
  [0.1, 0.25, 0.5, 1, 2, 4].map((v) => ({ value: v, label: v < 1 ? `${v}×`.replace("0.", ".") : `${v}×` })),
  ui.speed,
  (v) => (ui.speed = v),
);

window.addEventListener("keydown", (e) => {
  if (e.target instanceof HTMLInputElement && e.target.type !== "range") return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  switch (e.key) {
    case " ":
      e.preventDefault();
      if (!e.repeat) togglePlay();
      break;
    case "ArrowLeft":
      if (e.target instanceof HTMLInputElement) return;
      e.preventDefault();
      setRewinding(true);
      break;
    case "ArrowRight":
      if (e.target instanceof HTMLInputElement) return;
      e.preventDefault();
      stepForward();
      break;
    case "r":
    case "R":
      sim.reset();
      break;
    case "c":
    case "C":
      clearTrails();
      break;
    case "p":
    case "P":
      setPanel(!ui.panelOpen);
      break;
    default: {
      const i = Number(e.key) - 1;
      if (Number.isInteger(i) && i >= 0 && i < PRESETS.length) applyPreset(PRESETS[i]);
    }
  }
});
window.addEventListener("keyup", (e) => {
  if (e.key === "ArrowLeft") setRewinding(false);
});
window.addEventListener("blur", () => setRewinding(false));

// ---------------------------------------------------------------------------
// Direct manipulation: drag & fling the bobs
// ---------------------------------------------------------------------------
function bobScreenPositions(): { b1: [number, number]; b2: [number, number] } {
  const [x1, y1, x2, y2] = bobPositions(sim.state[0], sim.state[2], sim.params.L1, sim.params.L2);
  return { b1: renderer.toScreen(x1, y1), b2: renderer.toScreen(x2, y2) };
}

function hitTest(sx: number, sy: number): BobId | null {
  const { b1, b2 } = bobScreenPositions();
  const d1 = Math.hypot(sx - b1[0], sy - b1[1]);
  const d2 = Math.hypot(sx - b2[0], sy - b2[1]);
  const r1 = renderer.bobRadius(sim.params.m1) + 14;
  const r2 = renderer.bobRadius(sim.params.m2) + 14;
  if (d2 <= r2 && (d2 <= d1 || d1 > r1)) return "b2";
  if (d1 <= r1) return "b1";
  return null;
}

function localXY(e: PointerEvent): [number, number] {
  const r = canvas.getBoundingClientRect();
  return [e.clientX - r.left, e.clientY - r.top];
}

function dragTo(sx: number, sy: number): void {
  if (!drag) return;
  const [wx, wy] = renderer.toWorld(sx, sy);
  const { L1 } = sim.params;
  let th1 = drag.th1, th2 = drag.th2;
  if (drag.bob === "b1") {
    th1 = Math.atan2(wx, -wy);
  } else {
    const x1 = L1 * Math.sin(th1), y1 = -L1 * Math.cos(th1);
    th2 = Math.atan2(wx - x1, -(wy - y1));
  }
  // Keep angles continuous so velocities don't jump by 2π.
  drag.th1 = drag.th1 + wrapPi(th1 - drag.th1);
  drag.th2 = drag.th2 + wrapPi(th2 - drag.th2);
  const now = performance.now();
  drag.samples.push({ t: now, th1: drag.th1, th2: drag.th2 });
  while (drag.samples.length > 2 && now - drag.samples[0].t > 100) drag.samples.shift();
  sim.reseed({ th1: drag.th1, w1: 0, th2: drag.th2, w2: 0 });
}

canvas.addEventListener("pointerdown", (e) => {
  const [sx, sy] = localXY(e);
  const bob = hitTest(sx, sy);
  if (!bob) return;
  e.preventDefault();
  canvas.setPointerCapture(e.pointerId);
  const s = sim.state;
  drag = { bob, pointerId: e.pointerId, samples: [], th1: s[0], th2: s[2] };
  canvas.classList.add("grabbing");
  hint.classList.add("gone");
  markCustom();
  dragTo(sx, sy);
});

canvas.addEventListener("pointermove", (e) => {
  const [sx, sy] = localXY(e);
  if (drag && e.pointerId === drag.pointerId) {
    dragTo(sx, sy);
    return;
  }
  if (e.pointerType === "mouse") {
    ui.hover = hitTest(sx, sy);
    canvas.classList.toggle("can-grab", ui.hover !== null);
  }
});

function endDrag(e: PointerEvent): void {
  if (!drag || e.pointerId !== drag.pointerId) return;
  const now = performance.now();
  const s = drag.samples.filter((p) => now - p.t <= 100);
  let w1 = 0, w2 = 0;
  if (s.length >= 2) {
    const a = s[0], b = s[s.length - 1];
    const dt = (b.t - a.t) / 1000;
    if (dt > 0.008) {
      const clamp = (v: number) => Math.max(-25, Math.min(25, v));
      w1 = clamp((b.th1 - a.th1) / dt);
      w2 = clamp((b.th2 - a.th2) / dt);
    }
  }
  sim.reseed({ th1: drag.th1, w1, th2: drag.th2, w2 });
  drag = null;
  canvas.classList.remove("grabbing");
  syncPanel();
}
canvas.addEventListener("pointerup", endDrag);
canvas.addEventListener("pointercancel", endDrag);

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------
function layout(): void {
  renderer.resize();
  const W = window.innerWidth, H = window.innerHeight;
  const mobile = W <= 760;
  const gap = mobile ? 10 : 16;
  const top = gap + (mobile ? 52 : 56);
  if (mobile) {
    const bottom = ui.panelOpen ? H * 0.52 + gap : 0;
    renderer.setViewport({ x: 0, y: top + 40, w: W, h: Math.max(120, H - top - 40 - bottom) });
  } else {
    const right = ui.panelOpen ? 320 + gap * 2 : 0;
    renderer.setViewport({ x: 0, y: top, w: W - right, h: H - top });
  }
}
window.addEventListener("resize", layout);

// ---------------------------------------------------------------------------
// HUD
// ---------------------------------------------------------------------------
let fps = 60;
let hudTimer = 0;

function fmtAngle(a: number): string {
  return `${(wrapPi(a) * deg).toFixed(1).padStart(6)}°`;
}

function renderHud(): void {
  const s = sim.state;
  const status = drag ? ["holding", "paused"] : ui.rewinding ? ["rewind", "rewind"] : !ui.playing ? ["paused", "paused"] : sim.lagging ? ["slow", "lag"] : ["live", ""];
  const e = totalEnergy(s, 0, sim.params);
  const drift = sim.energyDrift();
  const driftTxt = sim.params.b > 0 ? "damped" : `${(drift * 100).toExponential(1)} %`;
  const driftCls = sim.params.b > 0 ? "" : Math.abs(drift) < 1e-4 ? "good" : "warn";
  hud.innerHTML = `
    <div class="big"><span>${sim.t.toFixed(2)}<small> s</small></span><span class="state ${status[1]}">${status[0]}</span></div>
    <span class="k">θ₁</span><span class="v">${fmtAngle(s[0])}</span>
    <span class="k">θ₂</span><span class="v">${fmtAngle(s[2])}</span>
    <span class="k extra">ω₁</span><span class="v extra">${s[1].toFixed(2)} rad/s</span>
    <span class="k extra">ω₂</span><span class="v extra">${s[3].toFixed(2)} rad/s</span>
    <span class="k">energy</span><span class="v">${e.toFixed(3)} J</span>
    <span class="k extra">drift</span><span class="v extra ${driftCls}">${driftTxt}</span>
    <span class="k extra">pendulums</span><span class="v extra">${sim.n}</span>
    <span class="k extra">rewind</span><span class="v extra">${sim.rewindSeconds.toFixed(1)} / ${sim.historyCapacitySeconds.toFixed(0)} s</span>
    <span class="k extra">fps</span><span class="v extra">${fps.toFixed(0)}</span>`;
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------
let last = performance.now();

function frame(now: number): void {
  const realDt = Math.min(0.05, (now - last) / 1000);
  last = now;
  fps += (1 / Math.max(realDt, 1e-3) - fps) * 0.05;

  if (!drag) {
    if (ui.rewinding) sim.rewind(realDt * ui.speed);
    else if (ui.playing) sim.advance(realDt * ui.speed);
  }

  renderer.draw(sim, {
    trailSeconds: ui.trailSeconds,
    trailMode: ui.trailMode,
    innerTrail: ui.innerTrail,
    showArms: ui.showArms,
    hover: ui.hover,
    dragging: drag?.bob ?? null,
  });

  if (sim.n > 1 && document.visibilityState === "visible") renderer.adaptQuality(realDt);

  hudTimer += realDt;
  if (hudTimer > 0.1) {
    hudTimer = 0;
    renderHud();
  }
  requestAnimationFrame(frame);
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
buildPanel();
applyPreset(first);
setPanel(ui.panelOpen);
setTimeout(() => hint.classList.add("gone"), 15000);
requestAnimationFrame(frame);

// Handy for poking at the simulation from the devtools console.
(window as unknown as { chaos: unknown }).chaos = { sim, ui, PRESETS, applyPreset, initialState: (): PendulumState => sim.seed };
