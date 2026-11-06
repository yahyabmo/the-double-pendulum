# Chaos Lab: double pendulum

A real-time, interactive double pendulum simulator that runs in the browser. You can grab the bobs and throw them, change gravity mid-swing, rewind time, and run a thousand pendulums side by side to watch the butterfly effect happen.

## Features

- **Real-time physics.** A fixed-step RK4 integrator runs at 600 steps per second by default. The HUD shows energy drift, which typically stays around 10⁻⁷ of the system's energy scale.
- **Direct manipulation.** Drag either bob to place it. Release it while moving to fling it with that speed.
- **Rewind.** Hold <kbd>←</kbd> or the rewind button to scrub back through the recorded history. This is exact, not re-integrated backwards.
- **Butterfly mode.** Run up to 2000 pendulums at once, each with its θ₂ nudged by a tiny Δθ (down to 10⁻¹²). They move as one, then burst into a rainbow fan.
- **Live physics.** Change gravity (with Moon, Mars, Earth and Jupiter presets), masses, lengths and joint damping while it runs.
- **Glowing trails.** Coloured by speed, by time (rainbow) or a single colour, built from the history buffer so they rewind too.
- **Presets.** Includes *The original*, which replays the setup from the first version of this project: m₁=2, m₂=1, L=2, 100 runs offset by 10⁻⁴.

### Keyboard

| Key | Action |
| --- | --- |
| <kbd>Space</kbd> | play / pause |
| hold <kbd>←</kbd> | rewind |
| <kbd>→</kbd> | step forward |
| <kbd>R</kbd> | reset to start |
| <kbd>C</kbd> | clear trails |
| <kbd>P</kbd> | toggle panel |
| <kbd>1</kbd>–<kbd>6</kbd> | presets |

## How it works

```
physics/derive.py   ──SymPy──▶  src/physics/generated.ts   (equations of motion)
                                        │
src/physics/sim.ts   RK4 ensemble integrator · history ring buffer · energy
src/render/renderer.ts  Canvas 2D renderer (trails, ensemble, glow)
src/ui/controls.ts      tiny dependency-free widgets
src/main.ts             app wiring, input, loop, HUD
```

The equations of motion are derived symbolically from the Lagrangian *L = T − V*, the same approach as the original script, which is kept in [`legacy/`](legacy/). Joint damping enters as a generalized force. The 2×2 mass-matrix system is solved symbolically, simplified with common-subexpression elimination, and emitted as TypeScript. To regenerate after changing the physics:

```bash
pip install sympy
npm run derive
```

The tests check the generated equations against the textbook closed form. They also check energy conservation, exact rewind, and that ensemble members diverge.

## Development

```bash
npm install
npm run dev      # http://localhost:5173
npm test
npm run build    # static site in dist/
```

## Deploying

`.github/workflows/deploy.yml` builds and publishes to GitHub Pages on every push to `main`. To enable it, go to **Settings → Pages → Build and deployment → Source** and choose **GitHub Actions**.

## Roadmap

- [x] **Phase 1.** Real-time simulator: drag and fling, live physics, rewind, trails, presets, butterfly ensemble.
- [ ] **Phase 2.** Chaos tools: divergence graph and Lyapunov exponent, phase-space and Poincaré plots, energy charts.
- [ ] **Phase 3.** GPU flip-time fractal map with zoom and click-to-launch.
- [ ] **Phase 4.** N-link chains, share-by-URL, GIF/video export, equations drawer.
- [ ] **Phase 5.** Balance game, swing-up autopilot, sonification, 3D spherical pendulum.
