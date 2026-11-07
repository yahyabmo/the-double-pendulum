# Chaos Lab: double pendulum

A real-time, interactive double pendulum simulator that runs in the browser. You can grab the bobs and throw them, change gravity mid-swing, rewind time, and run a thousand pendulums side by side to watch the butterfly effect happen.

## Installation

One command installs everything Chaos Lab needs, Node.js included, and puts a **Chaos Lab** shortcut on your desktop and in your applications menu.

**Linux**, in a terminal:

```bash
curl -fsSL https://raw.githubusercontent.com/yahyabmo/the-double-pendulum/main/install-linux.sh | bash
```

**Windows**, in CMD or PowerShell:

```bat
powershell -c "irm https://raw.githubusercontent.com/yahyabmo/the-double-pendulum/main/install-windows.bat | iex"
```

Rather not type a command? Download the project (green **Code** button, then **Download ZIP**) and unzip it. On Windows, double-click `install-windows.bat`. On Linux, right-click `install-linux.sh` and choose **Run as a Program**.

The first installation takes about a minute, then Chaos Lab opens in your browser.

## Launch

Double-click the **Chaos Lab** shortcut on your desktop, or run:

**Linux**, in a terminal:

```bash
~/.local/share/chaos-lab/launch.sh
```

**Windows**, in CMD:

```bat
"%LOCALAPPDATA%\chaos-lab\launch.cmd"
```

Chaos Lab opens at <http://localhost:5190>. Add `--stop` to the command to stop it.

### If something goes wrong

| Problem | Fix |
| --- | --- |
| Windows shows "Windows protected your PC" | Click **More info**, then **Run anyway**. |
| Linux offers no **Run as a Program** | Open a terminal in the folder and run `bash install-linux.sh` |
| On Ubuntu, the desktop shortcut does not start | Right-click it, then **Allow Launching**. |
| "Port 5190 is used by another program" | Close that program, then open Chaos Lab again. |
| The browser did not open | Open <http://localhost:5190> yourself. |

To uninstall Chaos Lab, run `~/.local/share/chaos-lab/uninstall.sh` on Linux, or `"%LOCALAPPDATA%\chaos-lab\uninstall.cmd"` in CMD on Windows.

## Screenshots

![A single chaotic pendulum with its glowing trail](docs/screenshots/chaos.png)

*A single pendulum in chaotic motion. The trail of the lower bob is coloured by speed, and the panel on the left shows angles, speeds and energy.*

![1000 pendulums splitting apart in butterfly mode](docs/screenshots/butterfly.png)

*Butterfly mode, 12 s in. 1000 pendulums started 10⁻⁷ rad apart: they moved as one, and now they are flying apart.*

![The setup of the original script, 100 pendulums](docs/screenshots/original.png)

*The original experiment of this project, live: m₁ = 2, m₂ = 1, L = 2, and 100 pendulums offset by 10⁻⁴ rad.*

<p align="center">
  <img src="docs/screenshots/mobile.png" alt="Chaos Lab on a phone" width="300">
</p>

<p align="center"><em>On a phone, the panel folds away and the controls stay at hand.</em></p>

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
npm run dev      # http://localhost:5190
npm test
npm run build    # static site in dist/
```

The site runs locally. On every push, `.github/workflows/ci.yml` runs the tests and the build, and `.github/workflows/installers.yml` installs Chaos Lab on Linux and on Windows, then uninstalls it.

## Roadmap

- [x] **Phase 1.** Real-time simulator: drag and fling, live physics, rewind, trails, presets, butterfly ensemble.
- [ ] **Phase 2.** Chaos tools: divergence graph and Lyapunov exponent, phase-space and Poincaré plots, energy charts.
- [ ] **Phase 3.** GPU flip-time fractal map with zoom and click-to-launch.
- [ ] **Phase 4.** N-link chains, share-by-URL, GIF/video export, equations drawer.
- [ ] **Phase 5.** Balance game, swing-up autopilot, sonification, 3D spherical pendulum.
