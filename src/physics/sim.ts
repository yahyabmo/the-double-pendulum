import { accelerations, kineticEnergy, potentialEnergy } from "./generated";

export interface Params {
  g: number;
  m1: number;
  m2: number;
  L1: number;
  L2: number;
  /** Joint damping coefficient (N·m·s). */
  b: number;
}

/** State of one pendulum: angles measured from the downward vertical (rad), angular velocities (rad/s). */
export interface PendulumState {
  th1: number;
  w1: number;
  th2: number;
  w2: number;
}

/** Number of floats per pendulum in packed state arrays: [th1, w1, th2, w2]. */
export const STRIDE = 4;

const acc = new Float64Array(2);

/**
 * Advance every pendulum in `s` (packed [th1,w1,th2,w2]*n) by one classic RK4 step of size dt.
 * Works in place with scalar locals only — no allocation in the hot loop.
 */
export function rk4Step(s: Float64Array, n: number, dt: number, p: Params): void {
  const { g, m1, m2, L1, L2, b } = p;
  const h2 = dt / 2;
  for (let i = 0; i < n; i++) {
    const o = i * STRIDE;
    const a = s[o],
      wa = s[o + 1],
      c = s[o + 2],
      wc = s[o + 3];

    accelerations(acc, a, c, wa, wc, g, m1, m2, L1, L2, b);
    const k1a = wa, k1wa = acc[0], k1c = wc, k1wc = acc[1];

    accelerations(acc, a + h2 * k1a, c + h2 * k1c, wa + h2 * k1wa, wc + h2 * k1wc, g, m1, m2, L1, L2, b);
    const k2a = wa + h2 * k1wa, k2wa = acc[0], k2c = wc + h2 * k1wc, k2wc = acc[1];

    accelerations(acc, a + h2 * k2a, c + h2 * k2c, wa + h2 * k2wa, wc + h2 * k2wc, g, m1, m2, L1, L2, b);
    const k3a = wa + h2 * k2wa, k3wa = acc[0], k3c = wc + h2 * k2wc, k3wc = acc[1];

    accelerations(acc, a + dt * k3a, c + dt * k3c, wa + dt * k3wa, wc + dt * k3wc, g, m1, m2, L1, L2, b);
    const k4a = wa + dt * k3wa, k4wa = acc[0], k4c = wc + dt * k3wc, k4wc = acc[1];

    const d6 = dt / 6;
    s[o] = a + d6 * (k1a + 2 * k2a + 2 * k3a + k4a);
    s[o + 1] = wa + d6 * (k1wa + 2 * k2wa + 2 * k3wa + k4wa);
    s[o + 2] = c + d6 * (k1c + 2 * k2c + 2 * k3c + k4c);
    s[o + 3] = wc + d6 * (k1wc + 2 * k2wc + 2 * k3wc + k4wc);
  }
}

export function totalEnergy(s: Float64Array, i: number, p: Params): number {
  const o = i * STRIDE;
  const th1 = s[o], w1 = s[o + 1], th2 = s[o + 2], w2 = s[o + 3];
  return (
    kineticEnergy(th1, th2, w1, w2, p.g, p.m1, p.m2, p.L1, p.L2) +
    potentialEnergy(th1, th2, w1, w2, p.g, p.m1, p.m2, p.L1, p.L2)
  );
}

/**
 * Ring buffer of past states sampled at a fixed simulation-time interval.
 * Powers exact rewind and history-derived trails.
 */
export class History {
  readonly capacity: number;
  readonly frameSize: number;
  private data: Float64Array;
  private times: Float64Array;
  private start = 0;
  length = 0;

  constructor(frameSize: number, capacity: number) {
    this.frameSize = frameSize;
    this.capacity = Math.max(2, capacity);
    this.data = new Float64Array(this.capacity * frameSize);
    this.times = new Float64Array(this.capacity);
  }

  clear(): void {
    this.start = 0;
    this.length = 0;
  }

  push(state: Float64Array, t: number): void {
    let idx: number;
    if (this.length < this.capacity) {
      idx = (this.start + this.length) % this.capacity;
      this.length++;
    } else {
      idx = this.start;
      this.start = (this.start + 1) % this.capacity;
    }
    this.data.set(state.subarray(0, this.frameSize), idx * this.frameSize);
    this.times[idx] = t;
  }

  /** Remove the newest frame. */
  pop(): void {
    if (this.length > 0) this.length--;
  }

  private physIndex(k: number): number {
    // k = 0 is the newest frame
    return (this.start + this.length - 1 - k + this.capacity) % this.capacity;
  }

  /** Offset into `buffer` of frame k (0 = newest). */
  offset(k: number): number {
    return this.physIndex(k) * this.frameSize;
  }

  time(k: number): number {
    return this.times[this.physIndex(k)];
  }

  get buffer(): Float64Array {
    return this.data;
  }

  copyInto(k: number, out: Float64Array): void {
    const o = this.offset(k);
    out.set(this.data.subarray(o, o + this.frameSize));
  }
}

export interface SimOptions {
  stepsPerSecond: number;
  /** Seconds of simulation time between history samples. */
  recordInterval: number;
  /** Max doubles the history may hold (bounds memory for large ensembles). */
  historyBudget: number;
  /** Max seconds of history to keep. */
  maxHistorySeconds: number;
}

const DEFAULT_OPTIONS: SimOptions = {
  stepsPerSecond: 600,
  recordInterval: 1 / 120,
  historyBudget: 6_000_000,
  maxHistorySeconds: 120,
};

/** A whole experiment: an ensemble of pendulums sharing parameters, with time, history and a seed. */
export class Simulation {
  params: Params;
  opts: SimOptions;
  n = 1;
  state = new Float64Array(STRIDE);
  t = 0;
  history!: History;
  /** Initial condition the ensemble was seeded from (pendulum 0). */
  seed: PendulumState = { th1: 0, w1: 0, th2: 0, w2: 0 };
  /** Perturbation on theta2 between consecutive ensemble members (rad). */
  spread = 1e-4;
  /** Energy of pendulum 0 at seed time (for drift measurement). */
  e0 = 0;

  private stepAccum = 0;
  private stepsSinceRecord = 0;
  /** Target time of an ongoing rewind (keeps sub-frame precision across calls). */
  private rewindCursor: number | null = null;
  /** True when the last frame couldn't keep up with real time. */
  lagging = false;

  constructor(params: Params, opts: Partial<SimOptions> = {}) {
    this.params = { ...params };
    this.opts = { ...DEFAULT_OPTIONS, ...opts };
    this.allocate(1);
  }

  get dt(): number {
    return 1 / this.opts.stepsPerSecond;
  }

  /** History is sampled every this many physics steps. */
  get recordEvery(): number {
    return Math.max(1, Math.round(this.opts.recordInterval * this.opts.stepsPerSecond));
  }

  /** Change integration accuracy without disturbing the motion. */
  setStepsPerSecond(sps: number): void {
    this.opts.stepsPerSecond = sps;
    this.stepsSinceRecord = Math.min(this.stepsSinceRecord, this.recordEvery - 1);
  }

  private allocate(n: number): void {
    this.n = n;
    this.state = new Float64Array(n * STRIDE);
    const frameSize = n * STRIDE;
    const cap = Math.min(
      Math.floor(this.opts.historyBudget / frameSize),
      Math.ceil(this.opts.maxHistorySeconds / this.opts.recordInterval),
    );
    this.history = new History(frameSize, cap);
  }

  /** Seconds of rewind currently available. */
  get rewindSeconds(): number {
    if (this.history.length < 2) return 0;
    return this.t - this.history.time(this.history.length - 1);
  }

  /** Seconds of history the buffer can hold at most. */
  get historyCapacitySeconds(): number {
    return this.history.capacity * this.opts.recordInterval;
  }

  /** (Re)seed the ensemble from a base state: member i gets theta2 + i*spread. */
  reseed(base: PendulumState, n = this.n, spread = this.spread, resetTime = true): void {
    if (n !== this.n) this.allocate(n);
    this.seed = { ...base };
    this.spread = spread;
    for (let i = 0; i < n; i++) {
      const o = i * STRIDE;
      this.state[o] = base.th1;
      this.state[o + 1] = base.w1;
      this.state[o + 2] = base.th2 + i * spread;
      this.state[o + 3] = base.w2;
    }
    if (resetTime) this.t = 0;
    this.history.clear();
    this.history.push(this.state, this.t);
    this.stepAccum = 0;
    this.stepsSinceRecord = 0;
    this.rewindCursor = null;
    this.e0 = totalEnergy(this.state, 0, this.params);
  }

  /** Reset to the last seed. */
  reset(): void {
    this.reseed(this.seed);
  }

  /** Re-baseline the energy-drift measurement (e.g. after a parameter change). */
  rebaseEnergy(): void {
    this.e0 = totalEnergy(this.state, 0, this.params);
  }

  get pendulum0(): PendulumState {
    const s = this.state;
    return { th1: s[0], w1: s[1], th2: s[2], w2: s[3] };
  }

  /** Single fixed step (used by "step" button and by advance). */
  stepOnce(): void {
    const dt = this.dt;
    rk4Step(this.state, this.n, dt, this.params);
    this.t += dt;
    this.rewindCursor = null;
    if (++this.stepsSinceRecord >= this.recordEvery) {
      this.stepsSinceRecord = 0;
      this.history.push(this.state, this.t);
    }
  }

  /**
   * Advance by `simSeconds` of simulation time using fixed steps.
   * Stops early if `budgetMs` of wall time is exceeded (sets `lagging`).
   */
  advance(simSeconds: number, budgetMs = 12): void {
    this.stepAccum += simSeconds;
    const dt = this.dt;
    const startT = performance.now();
    let steps = 0;
    this.lagging = false;
    while (this.stepAccum >= dt) {
      this.stepOnce();
      this.stepAccum -= dt;
      steps++;
      if ((steps & 7) === 0 && performance.now() - startT > budgetMs) {
        this.lagging = true;
        this.stepAccum = 0; // drop the backlog instead of spiralling
        break;
      }
    }
  }

  /** Rewind by `simSeconds` through recorded history. Returns false when no history is left. */
  rewind(simSeconds: number): boolean {
    const h = this.history;
    if (h.length === 0) return false;
    const target = (this.rewindCursor ?? this.t) - simSeconds;
    while (h.length > 1 && h.time(0) > target + 1e-9) h.pop();
    h.copyInto(0, this.state);
    this.t = h.time(0);
    this.rewindCursor = Math.max(target, this.t);
    this.stepAccum = 0;
    this.stepsSinceRecord = 0;
    return h.length > 1;
  }

  /** Pendulum-0 energy drift relative to the seed energy (fraction of the energy scale). */
  energyDrift(): number {
    const e = totalEnergy(this.state, 0, this.params);
    const p = this.params;
    // Normalise by a natural energy scale so drift is meaningful when e0 ≈ 0.
    const scale = Math.max(1e-9, (p.m1 + p.m2) * Math.abs(p.g) * (p.L1 + p.L2));
    return (e - this.e0) / scale;
  }
}

/** Positions of the two bobs (world units, y up, pivot at origin). */
export function bobPositions(th1: number, th2: number, L1: number, L2: number): [number, number, number, number] {
  const x1 = L1 * Math.sin(th1);
  const y1 = -L1 * Math.cos(th1);
  return [x1, y1, x1 + L2 * Math.sin(th2), y1 - L2 * Math.cos(th2)];
}
