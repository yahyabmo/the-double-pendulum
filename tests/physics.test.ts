import { describe, expect, it } from "vitest";
import { accelerations } from "../src/physics/generated";
import { Simulation, totalEnergy, type Params } from "../src/physics/sim";

const P: Params = { g: 9.81, m1: 2, m2: 1, L1: 2, L2: 2, b: 0 };

/** Textbook closed form (no damping), independent of the SymPy derivation. */
function reference(th1: number, th2: number, w1: number, w2: number, p: Params): [number, number] {
  const { g, m1, m2, L1, L2 } = p;
  const d = th1 - th2;
  const den = 2 * m1 + m2 - m2 * Math.cos(2 * d);
  const a1 =
    (-g * (2 * m1 + m2) * Math.sin(th1) -
      m2 * g * Math.sin(th1 - 2 * th2) -
      2 * Math.sin(d) * m2 * (w2 * w2 * L2 + w1 * w1 * L1 * Math.cos(d))) /
    (L1 * den);
  const a2 =
    (2 * Math.sin(d) * (w1 * w1 * L1 * (m1 + m2) + g * (m1 + m2) * Math.cos(th1) + w2 * w2 * L2 * m2 * Math.cos(d))) /
    (L2 * den);
  return [a1, a2];
}

describe("generated equations of motion", () => {
  it("match the textbook closed form", () => {
    const out = new Float64Array(2);
    let seed = 1;
    const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) * 2 - 1;
    for (let k = 0; k < 500; k++) {
      const p = { ...P, m1: 0.5 + Math.abs(rnd()) * 3, m2: 0.5 + Math.abs(rnd()) * 3, L1: 0.5 + Math.abs(rnd()) * 2 };
      const th1 = rnd() * 4, th2 = rnd() * 4, w1 = rnd() * 6, w2 = rnd() * 6;
      accelerations(out, th1, th2, w1, w2, p.g, p.m1, p.m2, p.L1, p.L2, 0);
      const [r1, r2] = reference(th1, th2, w1, w2, p);
      expect(out[0]).toBeCloseTo(r1, 9);
      expect(out[1]).toBeCloseTo(r2, 9);
    }
  });

  it("damping only removes energy", () => {
    const sim = new Simulation({ ...P, b: 0.5 });
    sim.reseed({ th1: 2, w1: 0, th2: 2.5, w2: 0 });
    let prev = totalEnergy(sim.state, 0, sim.params);
    for (let i = 0; i < 20; i++) {
      sim.advance(0.5, 1e9);
      const e = totalEnergy(sim.state, 0, sim.params);
      expect(e).toBeLessThanOrEqual(prev + 1e-6);
      prev = e;
    }
  });
});

describe("simulation", () => {
  it("conserves energy in chaotic motion (no damping)", () => {
    const sim = new Simulation(P, { stepsPerSecond: 600 });
    sim.reseed({ th1: -Math.PI, w1: 0, th2: -3.4, w2: 0 });
    sim.advance(30, 1e9);
    expect(Math.abs(sim.energyDrift())).toBeLessThan(1e-4);
  });

  it("rewinds exactly to recorded states", () => {
    const sim = new Simulation(P);
    sim.reseed({ th1: 1.5, w1: 0, th2: 2.5, w2: 0 });
    sim.advance(2, 1e9);
    const snapshotT = sim.history.time(0);
    const snapshot = new Float64Array(4);
    sim.history.copyInto(0, snapshot);
    sim.advance(3, 1e9);
    sim.rewind(sim.t - snapshotT);
    expect(sim.t).toBeCloseTo(snapshotT, 9);
    expect(Array.from(sim.state)).toEqual(Array.from(snapshot));
  });

  it("ensemble members diverge from tiny perturbations (butterfly effect)", () => {
    const sim = new Simulation(P);
    sim.reseed({ th1: -Math.PI, w1: 0, th2: -3.4, w2: 0 }, 2, 1e-6);
    sim.advance(30, 1e9);
    expect(Math.abs(sim.state[2] - sim.state[6])).toBeGreaterThan(0.1);
  });
});
