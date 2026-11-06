import type { Params, PendulumState } from "./physics/sim";

export interface Preset {
  id: string;
  name: string;
  blurb: string;
  params: Params;
  state: PendulumState;
  count: number;
  /** theta2 perturbation between ensemble members (rad). */
  spread: number;
  trailSeconds?: number;
}

const deg = Math.PI / 180;
const EARTH = { g: 9.81, m1: 1, m2: 1, L1: 1, L2: 1, b: 0 };

export const PRESETS: Preset[] = [
  {
    id: "chaos",
    name: "Chaos",
    blurb: "High energy, never repeats",
    params: EARTH,
    state: { th1: 125 * deg, w1: 0, th2: -40 * deg, w2: 0 },
    count: 1,
    spread: 1e-6,
    trailSeconds: 4,
  },
  {
    id: "butterfly",
    name: "Butterfly",
    blurb: "1000 pendulums, Δθ = 10⁻⁷",
    params: EARTH,
    state: { th1: 150 * deg, w1: 0, th2: 150 * deg, w2: 0 },
    count: 1000,
    spread: 1e-7,
    trailSeconds: 0.4,
  },
  {
    id: "original",
    name: "The original",
    blurb: "Your first script: 100 runs",
    params: { g: 9.81, m1: 2, m2: 1, L1: 2, L2: 2, b: 0 },
    state: { th1: -1.5708 * 2, w1: 0, th2: -1.7 * 2, w2: 0 },
    count: 100,
    spread: -1e-4,
    trailSeconds: 1,
  },
  {
    id: "gentle",
    name: "Gentle",
    blurb: "Small angles, near-periodic",
    params: EARTH,
    state: { th1: 22 * deg, w1: 0, th2: 30 * deg, w2: 0 },
    count: 1,
    spread: 1e-6,
    trailSeconds: 6,
  },
  {
    id: "balance",
    name: "Balanced",
    blurb: "Upside down, 10⁻⁹ off",
    params: EARTH,
    state: { th1: Math.PI, w1: 0, th2: Math.PI + 1e-9, w2: 0 },
    count: 1,
    spread: 1e-6,
    trailSeconds: 5,
  },
  {
    id: "moon",
    name: "Moon",
    blurb: "g = 1.62, slow-motion chaos",
    params: { ...EARTH, g: 1.62 },
    state: { th1: 100 * deg, w1: 0, th2: 160 * deg, w2: 0 },
    count: 1,
    spread: 1e-6,
    trailSeconds: 8,
  },
];

export const PLANETS: { label: string; g: number }[] = [
  { label: "Moon", g: 1.62 },
  { label: "Mars", g: 3.71 },
  { label: "Earth", g: 9.81 },
  { label: "Jupiter", g: 24.79 },
];
