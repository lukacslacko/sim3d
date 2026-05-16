# sim3d

Single-file physics-style simulators in HTML + JS, exploring self-organising
fields on a periodic 3D torus. Motivation: compact field models that produce
closed worldlines and channel-like structures without a globally distinguished
time direction (in the spirit of Greg Egan's *Orthogonal* trilogy).

Everything visual is a static HTML page — no build step. WebGL2 is used for
GPU compute (fragment-shader passes over an `Nt×Nt` atlas of `N×N` slices).
THREE.js is pulled from a CDN via `<script type="importmap">` where needed.

## Running

Serve the directory over HTTP — browsers block ES-module `import` (and the
sibling `.mjs` files used by `align.html` / `viewer.html` / `index.html`)
when pages are opened as `file://`:

```sh
cd /path/to/sim3d
python3 -m http.server 8000
# then visit http://localhost:8000/flow3d.html  (etc.)
```

Any static server works — `python3 -m http.server`, `npx serve`, etc.
The self-contained pages (`flow3d.html`, `flow3d_tr.html`, `grav2d.html`,
`grad3d.html`) can also be opened directly with `open <name>.html`, but
serving is the uniform path.

WebGL2 + `EXT_color_buffer_float` is required for the GPU pages (any
recent Chrome / Firefox / Safari).

## Simulators (HTML)

- `flow3d.html` — sourceless mass-flow `V` with self-gravity on the 3D torus.
  Gradient descent on `E = ∫ [½|∇V|² − Φ|V| + (1/8πG)|∇Φ|²]` with `∇·V = 0`
  and `∇²Φ = −κ|V|`. Helmholtz projection, optional mass-rescale, refine ×2.
- `flow3d_tr.html` — same energy, decomposed `V = ρ·T` with `|T| = 1`
  (unit time-direction `T : T³ → S²`, density `ρ`). Periodic-torus or cube
  (Dirichlet `ρ=Φ=0` outside, `T` = unit radial-outward from cube centre).
- `align.html` — `N³` lattice of orientations on the torus relaxed by
  averaging each cell toward its 6 face-adjacent neighbours. Two value
  modes: `so3` (unit quaternions, `π₁(SO(3))=ℤ/2` disclination "worms")
  and `s1` (angles in `[0, 2π)`). Includes voxel render, rotation-vector
  ball, parallel-transport tubes, observer worldlines.
- `viewer.html` — read-only viewer for `.qsnap` snapshots produced by
  `align_gpu.py` (or by saving from `align.html`).
- `grav2d.html` — 2D version of the mass-flow / prescribed-curl model with
  literal kernel sums (cutoff disk, smoothstep window) on a periodic torus.
- `grad3d.html` — 3D `M / ω` prescribed-curl variant, Biot–Savart-style.
  Kept for history; not in active development.
- `index.html` — earlier block-affinity playground (sheet / head-tail /
  head-body-tail / head-tail-water configurations) that uses `physics.mjs`.

## Supporting JS modules

Loaded by the HTML pages above.

- `physics.mjs` — shared affinity / bonding physics for `index.html` and
  the headless checks. Pure JS, also runs under node.
- `align.mjs` — CPU step for the orientation-alignment grid (`so3` / `s1`).
- `align_gpu.mjs` — WebGL2 implementation of the same step, ping-ponging
  two `RGBA32F` render targets.
- `histogram.mjs` — spherical direction histogram on a subdivided
  icosahedron, used by `index.html` / `align.html` / `viewer.html`.

## Headless CLI scripts (node)

Run with plain node — no `npm install` needed; they only import sibling
`.mjs` files.

```sh
node simulate.mjs                # block-affinity run, default new model
node simulate.mjs --model=old --steps=4000 --n=80 --R=8 --seed=1
node grad_check.mjs              # finite-difference gradient sanity check
node honeycomb_check.mjs         # hex-layer / HBT chain analysis
```

## Python: GPU SO(3) alignment

`align_gpu.py` runs the alignment iteration on Apple-silicon GPUs via
`jax-metal`. A pre-built venv is committed under `venv/`.

```sh
source venv/bin/activate
python align_gpu.py 128
# default: full size chain 16 → 32 → 64 → 128 with 10s warm / 10s ramp / 10s hold

python align_gpu.py 64 0.05                                # custom initial noise
python align_gpu.py 128 --start-size 32 --init align_32.qsnap   # resume from 32
python align_gpu.py 32 0 --start-size 32 --warmup-time 0 --ramp-time 0 --hold-time 30
```

Each stage writes a `.qsnap` snapshot. Format:

```
[u64 LE  : header length]
[bytes   : JSON metadata {N, cubeSize, eta, noise, steps, mode}]
[N³ × 4 × float32 LE : quaternions (x, y, z, w), cell-major (iz, iy, ix)]
```

Open snapshots with `viewer.html` or load them into `align.html`. Loading
a snapshot of half the requested `N` upsamples 2× by nearest-neighbour copy
(same convention as `align.html`'s **Refine ×2**).

## Conventions

- One file per simulator. New ones live as `<name>.html`; no per-feature
  build pipelines.
- WebGL2 + `EXT_color_buffer_float` for GPU compute. CPU JS for visuals
  (slices, streamlines, histograms).
- Discretisation: backward differences for div/curl, forward differences
  for grad and `curl_fwd`, so the implied Laplacian is the standard
  5/7-point stencil.
