import { STRIDE, Simulation } from "../physics/sim";

export type TrailMode = "speed" | "rainbow" | "solid";
export type BobId = "b1" | "b2";

export interface DrawOptions {
  trailSeconds: number;
  trailMode: TrailMode;
  innerTrail: boolean;
  showArms: boolean;
  hover: BobId | null;
  dragging: BobId | null;
}

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** Ensemble colour for member i of n (a "jet"-like rainbow, as in the original script). */
export function ensembleHue(i: number, n: number): number {
  return n <= 1 ? 200 : 240 - (300 * i) / (n - 1);
}

function makeGlowSprite(color: string, size = 128): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const g = c.getContext("2d")!;
  const grd = g.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  grd.addColorStop(0, color);
  grd.addColorStop(0.25, color.replace(/[\d.]+\)$/, "0.35)"));
  grd.addColorStop(1, color.replace(/[\d.]+\)$/, "0)"));
  g.fillStyle = grd;
  g.fillRect(0, 0, size, size);
  return c;
}

export class Renderer {
  readonly canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private dpr = 1;
  private cssW = 0;
  private cssH = 0;
  private view: Rect = { x: 0, y: 0, w: 0, h: 0 };
  private viewTarget: Rect | null = null;
  /** Pixels per metre, eased towards its target for smooth zooming when lengths change. */
  private scale = 0;
  private glowA = makeGlowSprite("rgba(125, 211, 252, 1)");
  private glowB = makeGlowSprite("rgba(244, 114, 182, 1)");
  /** Scratch buffer of trail points [x, y, t] (reused between frames). */
  private trailScratch = new Float64Array(0);
  /** Total ensemble-trail points per frame; adapted to keep the frame rate up. */
  trailBudget = 24_000;
  private perfWindow = { frames: 0, time: 0 };

  /** Call once per frame with the real frame time; nudges quality up or down. */
  adaptQuality(frameSeconds: number): void {
    const w = this.perfWindow;
    w.frames++;
    w.time += frameSeconds;
    if (w.time < 0.75) return;
    const fps = w.frames / w.time;
    if (fps < 50) this.trailBudget = Math.max(2_000, this.trailBudget * 0.6);
    else if (fps > 57) this.trailBudget = Math.min(40_000, this.trailBudget * 1.15);
    w.frames = 0;
    w.time = 0;
  }

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d", { alpha: false })!;
  }

  resize(): void {
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    const r = this.canvas.getBoundingClientRect();
    this.cssW = r.width;
    this.cssH = r.height;
    this.canvas.width = Math.round(r.width * this.dpr);
    this.canvas.height = Math.round(r.height * this.dpr);
  }

  /** The unobstructed region of the canvas (CSS px) to centre the pendulum in. */
  setViewport(rect: Rect): void {
    if (!this.viewTarget) this.view = { ...rect };
    this.viewTarget = { ...rect };
  }

  /** Ease the view towards its target so panel toggles glide instead of jump. */
  private easeView(): void {
    const t = this.viewTarget;
    if (!t) return;
    const v = this.view;
    const k = 0.2;
    v.x += (t.x - v.x) * k;
    v.y += (t.y - v.y) * k;
    v.w += (t.w - v.w) * k;
    v.h += (t.h - v.h) * k;
  }

  private targetScale(reach: number): number {
    const v = this.view;
    return (0.43 * Math.min(v.w, v.h)) / Math.max(reach, 1e-6);
  }

  get pivot(): [number, number] {
    const v = this.view;
    return [v.x + v.w / 2, v.y + v.h / 2];
  }

  /** World (m, y up) → screen (CSS px). */
  toScreen(x: number, y: number): [number, number] {
    const [px, py] = this.pivot;
    return [px + x * this.scale, py - y * this.scale];
  }

  /** Screen (CSS px) → world (m, y up). */
  toWorld(sx: number, sy: number): [number, number] {
    const [px, py] = this.pivot;
    return [(sx - px) / this.scale, -(sy - py) / this.scale];
  }

  bobRadius(mass: number): number {
    return 7 + 6 * Math.cbrt(mass);
  }

  draw(sim: Simulation, o: DrawOptions): void {
    const ctx = this.ctx;
    const p = sim.params;
    const reach = p.L1 + p.L2;
    this.easeView();
    const target = this.targetScale(reach);
    this.scale = this.scale === 0 ? target : this.scale + (target - this.scale) * 0.18;

    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    this.drawBackground(reach);

    const many = sim.n > 1;
    if (o.trailSeconds > 0) {
      if (many) this.drawEnsembleTrails(sim, o);
      else this.drawTrail(sim, o, 2);
      if (!many && o.innerTrail) this.drawTrail(sim, o, 1);
    }
    if (many) this.drawEnsembleBodies(sim, o);
    this.drawLeader(sim, o, many);
  }

  private drawBackground(reach: number): void {
    const ctx = this.ctx;
    const w = this.cssW, h = this.cssH;
    const [px, py] = this.pivot;
    const bg = ctx.createRadialGradient(px, py, 0, px, py, Math.hypot(w, h) * 0.7);
    bg.addColorStop(0, "#0f1424");
    bg.addColorStop(1, "#05070d");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, w, h);

    // dot grid every 0.5 m
    const step = 0.5 * this.scale;
    if (step > 8) {
      ctx.fillStyle = "rgba(148, 163, 184, 0.07)";
      const x0 = px - Math.ceil(px / step) * step;
      const y0 = py - Math.ceil(py / step) * step;
      for (let x = x0; x < w; x += step)
        for (let y = y0; y < h; y += step) ctx.fillRect(x - 0.75, y - 0.75, 1.5, 1.5);
    }

    // reach circle
    ctx.strokeStyle = "rgba(148, 163, 184, 0.10)";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 6]);
    ctx.beginPath();
    ctx.arc(px, py, reach * this.scale, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  /** Number of history frames to use for a trail of the requested length. */
  private trailFrames(sim: Simulation, seconds: number, perPendulumBudget = Infinity): number {
    const wanted = Math.ceil(seconds / sim.opts.recordInterval);
    return Math.max(0, Math.min(wanted, sim.history.length, perPendulumBudget));
  }

  /** Fading, speed-coloured trail of pendulum 0's bob (1 = inner, 2 = outer). */
  private drawTrail(sim: Simulation, o: DrawOptions, bob: 1 | 2): void {
    const ctx = this.ctx;
    const h = sim.history;
    const { L1, L2 } = sim.params;
    const K = this.trailFrames(sim, o.trailSeconds);
    if (K < 2) return;

    // Gather screen points oldest → newest (+ live head).
    const pts = K + 1;
    if (this.trailScratch.length < pts * 3) this.trailScratch = new Float64Array(pts * 3);
    const buf = h.buffer;
    const [px, py] = this.pivot;
    const sc = this.scale;
    const write = (j: number, th1: number, th2: number, t: number) => {
      let x = L1 * Math.sin(th1);
      let y = -L1 * Math.cos(th1);
      if (bob === 2) {
        x += L2 * Math.sin(th2);
        y -= L2 * Math.cos(th2);
      }
      this.trailScratch[j * 3] = px + x * sc;
      this.trailScratch[j * 3 + 1] = py - y * sc;
      this.trailScratch[j * 3 + 2] = t;
    };
    for (let k = K - 1, j = 0; k >= 0; k--, j++) {
      const off = h.offset(k);
      write(j, buf[off], buf[off + 2], h.time(k));
    }
    write(K, sim.state[0], sim.state[2], sim.t);

    const P = this.trailScratch;
    const vScale = 1 / (Math.sqrt(Math.abs(sim.params.g) * (L1 + L2)) * 2.2 * sc + 1e-9);
    ctx.globalCompositeOperation = "lighter";
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    const dim = bob === 1 ? 0.45 : 1;
    const styleOf = (j: number): number => {
      // Quantized (hue, alpha) so neighbouring segments share one polyline (no beading at joints).
      const age = j / (pts - 1);
      const aq = Math.round(age * age * 24); // 0..24
      let hue: number;
      if (o.trailMode === "speed") {
        const dtp = Math.max(1e-6, P[j * 3 + 2] - P[(j - 1) * 3 + 2]);
        const v = Math.hypot(P[j * 3] - P[(j - 1) * 3], P[j * 3 + 1] - P[(j - 1) * 3 + 1]) / dtp;
        const u = Math.min(1, v * vScale);
        hue = 195 + 205 * Math.pow(u, 1.4);
      } else if (o.trailMode === "rainbow") {
        hue = P[j * 3 + 2] * 40;
      } else {
        hue = 195;
      }
      return aq * 1000 + (Math.round(hue / 6) % 60);
    };
    const styles = new Int32Array(pts);
    for (let j = 1; j < pts; j++) styles[j] = styleOf(j);
    for (let pass = 0; pass < 2; pass++) {
      ctx.lineWidth = pass === 0 ? (bob === 2 ? 7 : 5) : bob === 2 ? 2.2 : 1.6;
      let j = 1;
      while (j < pts) {
        const st = styles[j];
        const aq = Math.floor(st / 1000);
        const hue = (st % 1000) * 6;
        const a = (aq / 24) * dim * (pass === 0 ? 0.12 : 0.9);
        ctx.beginPath();
        ctx.moveTo(P[(j - 1) * 3], P[(j - 1) * 3 + 1]);
        while (j < pts && styles[j] === st) {
          ctx.lineTo(P[j * 3], P[j * 3 + 1]);
          j++;
        }
        if (a < 0.004) continue;
        ctx.strokeStyle = `hsla(${hue}, 95%, 64%, ${a})`;
        ctx.stroke();
      }
    }
    ctx.globalCompositeOperation = "source-over";
  }

  /** Pendulums are drawn in colour buckets: one path per bucket instead of one per pendulum. */
  private buckets(n: number): number {
    return Math.min(n, 40);
  }

  private drawEnsembleTrails(sim: Simulation, o: DrawOptions): void {
    const ctx = this.ctx;
    const h = sim.history;
    const n = sim.n;
    const { L1, L2 } = sim.params;
    const frames = this.trailFrames(sim, o.trailSeconds);
    if (frames < 2) return;
    // Keep total path points bounded: sample history sparsely for big ensembles.
    const maxPts = Math.max(3, Math.floor(this.trailBudget / n));
    const stride = Math.max(1, Math.ceil(frames / maxPts));
    const K = Math.floor((frames - 1) / stride) + 1;
    const buf = h.buffer;
    const [px, py] = this.pivot;
    const sc = this.scale;
    const offs = new Int32Array(K);
    for (let j = 0; j < K; j++) offs[j] = h.offset((K - 1 - j) * stride);

    const B = this.buckets(n);
    ctx.globalCompositeOperation = "lighter";
    ctx.lineWidth = n > 200 ? 1 : 1.4;
    ctx.lineJoin = "bevel";
    const alpha = Math.min(0.6, Math.max(0.12, 3 / Math.sqrt(n)));
    for (let bk = 0; bk < B; bk++) {
      const i0 = Math.floor((bk * n) / B), i1 = Math.floor(((bk + 1) * n) / B);
      ctx.beginPath();
      for (let i = i0; i < i1; i++) {
        for (let j = 0; j < K; j++) {
          const s = offs[j] + i * STRIDE;
          const th1 = buf[s], th2 = buf[s + 2];
          const x = px + (L1 * Math.sin(th1) + L2 * Math.sin(th2)) * sc;
          const y = py + (L1 * Math.cos(th1) + L2 * Math.cos(th2)) * sc;
          if (j === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        const s = i * STRIDE;
        ctx.lineTo(
          px + (L1 * Math.sin(sim.state[s]) + L2 * Math.sin(sim.state[s + 2])) * sc,
          py + (L1 * Math.cos(sim.state[s]) + L2 * Math.cos(sim.state[s + 2])) * sc,
        );
      }
      ctx.strokeStyle = `hsla(${ensembleHue((i0 + i1 - 1) / 2, n)}, 95%, 60%, ${alpha})`;
      ctx.stroke();
    }
    ctx.globalCompositeOperation = "source-over";
  }

  private drawEnsembleBodies(sim: Simulation, o: DrawOptions): void {
    const ctx = this.ctx;
    const n = sim.n;
    const { L1, L2 } = sim.params;
    const [px, py] = this.pivot;
    const sc = this.scale;
    const s = sim.state;
    const B = this.buckets(n);
    const armAlpha = Math.min(0.7, Math.max(0.18, 3 / Math.sqrt(n)));
    const r = n > 400 ? 1.6 : 2.6;
    const dots: number[] = [];
    ctx.globalCompositeOperation = "lighter";
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = n > 200 ? 1.2 : 2;
    for (let bk = B - 1; bk >= 0; bk--) {
      const i0 = Math.max(1, Math.floor((bk * n) / B)), i1 = Math.floor(((bk + 1) * n) / B);
      if (i1 <= i0) continue;
      const hue = ensembleHue((i0 + i1 - 1) / 2, n);
      ctx.beginPath();
      for (let i = i0; i < i1; i++) {
        const o4 = i * STRIDE;
        const x1 = px + L1 * Math.sin(s[o4]) * sc;
        const y1 = py + L1 * Math.cos(s[o4]) * sc;
        const x2 = x1 + L2 * Math.sin(s[o4 + 2]) * sc;
        const y2 = y1 + L2 * Math.cos(s[o4 + 2]) * sc;
        if (o.showArms) {
          ctx.moveTo(px, py);
          ctx.lineTo(x1, y1);
          ctx.lineTo(x2, y2);
        }
        dots.push(x2, y2);
      }
      if (o.showArms) {
        ctx.strokeStyle = `hsla(${hue}, 90%, 62%, ${armAlpha})`;
        ctx.stroke();
      }
      ctx.fillStyle = `hsla(${hue}, 95%, 68%, 0.9)`;
      ctx.beginPath();
      for (let k = 0; k < dots.length; k += 2) ctx.rect(dots[k] - r, dots[k + 1] - r, r * 2, r * 2);
      ctx.fill();
      dots.length = 0;
    }
    ctx.globalCompositeOperation = "source-over";
  }

  /** Pendulum 0: the one you can grab. */
  private drawLeader(sim: Simulation, o: DrawOptions, many: boolean): void {
    const ctx = this.ctx;
    const { L1, L2, m1, m2 } = sim.params;
    const [px, py] = this.pivot;
    const sc = this.scale;
    const s = sim.state;
    const x1 = px + L1 * Math.sin(s[0]) * sc;
    const y1 = py + L1 * Math.cos(s[0]) * sc;
    const x2 = x1 + L2 * Math.sin(s[2]) * sc;
    const y2 = y1 + L2 * Math.cos(s[2]) * sc;
    const r1 = this.bobRadius(m1);
    const r2 = this.bobRadius(m2);

    if (o.showArms || !many) {
      ctx.lineCap = "round";
      ctx.strokeStyle = "rgba(226, 232, 240, 0.85)";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }

    // pivot mount
    ctx.fillStyle = "#cbd5e1";
    ctx.beginPath();
    ctx.arc(px, py, 4.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(203, 213, 225, 0.35)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(px, py, 9, 0, Math.PI * 2);
    ctx.stroke();

    ctx.globalCompositeOperation = "lighter";
    const g1 = r1 * 5, g2 = r2 * 6;
    ctx.globalAlpha = 0.55;
    ctx.drawImage(this.glowA, x1 - g1 / 2, y1 - g1 / 2, g1, g1);
    ctx.globalAlpha = 0.75;
    ctx.drawImage(this.glowB, x2 - g2 / 2, y2 - g2 / 2, g2, g2);
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = "source-over";

    this.drawBob(x1, y1, r1, "#7dd3fc", "#0ea5e9", o.hover === "b1" || o.dragging === "b1");
    this.drawBob(x2, y2, r2, "#f9a8d4", "#db2777", o.hover === "b2" || o.dragging === "b2");
  }

  private drawBob(x: number, y: number, r: number, light: string, dark: string, active: boolean): void {
    const ctx = this.ctx;
    const grd = ctx.createRadialGradient(x - r * 0.35, y - r * 0.4, r * 0.1, x, y, r);
    grd.addColorStop(0, "#ffffff");
    grd.addColorStop(0.35, light);
    grd.addColorStop(1, dark);
    ctx.fillStyle = grd;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
    if (active) {
      ctx.strokeStyle = "rgba(255,255,255,0.85)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, r + 6, 0, Math.PI * 2);
      ctx.stroke();
    }
  }
}
