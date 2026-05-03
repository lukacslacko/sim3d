#!/usr/bin/env node
// Honeycomb / hex-layer analysis for HBT chains.
//
// 1. Compute the geometric NN distances for a 2D hex layer of HBT chains.
// 2. For each pair type, given a target equilibrium distance d_min plus a
//    chosen repulsion + attraction strength, numerically solve for the
//    'dist' parameter that puts the minimum of U = -rep/d + att·W exactly
//    at d_min.    (The minimum sits at d_min ≠ dist because the -rep/d term
//    is monotone, so it shifts the well's bottom outward.)
// 3. Verify the chosen tuning has hard-core repulsion (U → +∞ as d → 0).
// 4. Build a 19-chain hex patch and confirm forces are zero at the lattice
//    and that perturbations are restoring.

import {
  DEFAULTS, makeBlock, refreshNormal, applyPair,
} from './physics.mjs';

// ────────────────────────────────────────────────────────────────────────────
// 1. Geometry
// ────────────────────────────────────────────────────────────────────────────
const MOLECULE_LENGTH    = 0.3;
const INTERMOLECULE_DIST = 0.3;
const BOND               = MOLECULE_LENGTH / 2;            // 0.15
const D_HH = INTERMOLECULE_DIST;                           // 0.30
const D_BB = INTERMOLECULE_DIST;
const D_TT = INTERMOLECULE_DIST;
const D_HB = Math.hypot(INTERMOLECULE_DIST, BOND);         // 0.33541
const D_BT = D_HB;
const D_HT = MOLECULE_LENGTH;                              // 0.30 (intra-chain only)

console.log('── Geometric NN distances for hex layer ──');
console.log(`  bond_distance    = ${BOND.toFixed(5)}    (chain length = ${MOLECULE_LENGTH})`);
console.log(`  HH / BB / TT     = ${D_HH.toFixed(5)}    (in-layer NN)`);
console.log(`  HB / BT          = ${D_HB.toFixed(5)}    (= √(${INTERMOLECULE_DIST}² + ${BOND}²))`);
console.log(`  HT (intra-chain) = ${D_HT.toFixed(5)}`);
console.log();

// ────────────────────────────────────────────────────────────────────────────
// 2. Pair-energy tools.
//
// U(d) = -rep/d + att / (1 + (d − dist)²)              (cf=1 at full alignment)
// dU/dd = +rep/d² - 2(d − dist) · att / (1 + (d − dist)²)²
//
// At a stable minimum d_min: dU/dd = 0, i.e.
//   |rep| / d_min² = 2 (d_min − dist) · |att| / (1 + (d_min − dist)²)²
// Define f(δ) = δ / (1 + δ²)². This f peaks at δ = 1/√3 with max 3√3/16
// ≈ 0.3248. So a solution exists iff |rep| / d_min² ≤ 2·|att|·0.3248,
// i.e. |att| ≥ |rep| / (0.6495 · d_min²).
// ────────────────────────────────────────────────────────────────────────────
function pairU(d, rep, att, dist) {
  const dpk = d - dist;
  return -rep / d + att / (1 + dpk * dpk);
}
function pairDUdd(d, rep, att, dist) {
  const dpk = d - dist;
  const W = 1 / (1 + dpk * dpk);
  return rep / (d * d) - 2 * dpk * att * W * W;
}

// Solve for dist such that pairDUdd(d_min, rep, att, dist) = 0.
// Equivalent to: f(δ) = R, where δ = d_min − dist, R = |rep| / (2·|att|·d_min²),
// f(δ) = δ/(1+δ²)². We pick the SMALLER root (the one with δ ≤ 1/√3) so
// d_min IS a stable minimum (not the unstable "barrier-top" branch).
function solveDist(d_min, rep, att) {
  const aRep = Math.abs(rep);
  const aAtt = Math.abs(att);
  if (aRep === 0) return d_min;          // rep=0 → minimum at dist exactly
  const R = aRep / (2 * aAtt * d_min * d_min);
  const fmax = 3 * Math.sqrt(3) / 16;
  if (R > fmax + 1e-12) return null;     // no solution: rep too strong / att too weak
  // Bisect on δ ∈ [0, 1/√3]
  let lo = 0, hi = 1 / Math.sqrt(3);
  for (let i = 0; i < 80; i++) {
    const mid = 0.5 * (lo + hi);
    const f = mid / (1 + mid * mid) ** 2;
    if (f < R) lo = mid; else hi = mid;
  }
  const delta = 0.5 * (lo + hi);
  return d_min - delta;
}

// Numerical spring constant at d=d_min.
function springAt(d_min, rep, att, dist) {
  const h = 1e-3;
  const Up = pairU(d_min + h, rep, att, dist);
  const Um = pairU(d_min - h, rep, att, dist);
  const U0 = pairU(d_min,     rep, att, dist);
  return (Up + Um - 2 * U0) / (h * h);
}

// Barrier height (energy at d=ε relative to U(d_min)). Positive ⇒ hard-ish core.
function coreBarrier(d_min, rep, att, dist, eps = 0.02) {
  return pairU(eps, rep, att, dist) - pairU(d_min, rep, att, dist);
}

// ────────────────────────────────────────────────────────────────────────────
// 3. Pick |rep| and |att| per pair type, solve for dist.
//
// Constraint: |att| ≥ |rep| / (0.6495 · d_min²) for a valid minimum to exist.
// Larger |rep|  → stronger core barrier (good against clumping).
// Larger |att|  → deeper well & stiffer spring at the equilibrium.
//
// In-layer pairs (HH, BB, TT) are the structural skeleton — generous well
// + strong core. Cross-layer (HB, BT) are weaker (bonds do most of the work
// holding layers together). HT is intra-chain only and given a small core
// repulsion to encourage chain extension without overwhelming the bond.
// ────────────────────────────────────────────────────────────────────────────
// Mirror the structure of the user's previously-working tuning at the larger
// scale: same-type pairs get a real well (att<0); cross-type pairs (HB, BT)
// get pure core repulsion (att=0) — the bond holds layers together, and
// pure-rep cross-layer prevents both clumping and the chain-compression
// bias the long-range W tail of an attractive cross-layer well would cause.
// Intra-chain HT gets nothing — bonds set the chain length exactly.
const tunings = [
  { name: 'HH', d_min: D_HH, rep: -0.025, att: -1 },
  { name: 'BB', d_min: D_BB, rep: -0.025, att: -1 },
  { name: 'TT', d_min: D_TT, rep: -0.025, att: -1 },
  { name: 'HB', d_min: D_HB, rep: -0.025, att:  0 },
  { name: 'BT', d_min: D_BT, rep: -0.025, att:  0 },
  { name: 'HT', d_min: D_HT, rep:  0,     att:  0 },
];

console.log('── Pair tunings (computed `dist` puts minimum exactly at d_min) ──');
console.log('  pair  d_min     rep       att   →   dist      k_pair  T_pair  U_well_depth  core_barrier(ε=0.02)');
const solved = [];
for (const t of tunings) {
  // Special case: att=0 is pure -rep/d, no well. Just report.
  if (t.att === 0) {
    const dist = t.d_min;
    const k = 0;  // no well, no spring
    const barrier = -t.rep / 0.02 - (-t.rep / t.d_min);
    console.log(`  ${t.name}    ${t.d_min.toFixed(5)}  ${t.rep.toFixed(3)}    ${t.att.toFixed(2)}  → ${dist.toFixed(5)}  (no well — pure core repulsion)`);
    solved.push({ ...t, dist });
    continue;
  }
  const dist = solveDist(t.d_min, t.rep, t.att);
  if (dist === null) {
    console.log(`  ${t.name}: NO SOLUTION (|rep|=${Math.abs(t.rep)} is too strong vs |att|=${Math.abs(t.att)} at d_min=${t.d_min})`);
    continue;
  }
  const k = springAt(t.d_min, t.rep, t.att, dist);
  const T = k > 0 ? 2 * Math.PI / Math.sqrt(k) : Infinity;
  const wellDepth = pairU(Infinity, t.rep, t.att, dist) - pairU(t.d_min, t.rep, t.att, dist);
  const barrier = coreBarrier(t.d_min, t.rep, t.att, dist);
  console.log(
    `  ${t.name}    ${t.d_min.toFixed(5)}  ${t.rep.toFixed(3)}     ${t.att.toFixed(2)}  → ${dist.toFixed(5)}    ${k.toFixed(2)}    ${T.toFixed(2)}    ${(-wellDepth).toFixed(3)}        ${barrier.toFixed(2)}`
  );
  solved.push({ ...t, dist });
}
console.log();
console.log('  k_pair = d²U/dd² at d=d_min (per pair, per unit displacement²)');
console.log('  T_pair = 2π/√k_pair (single-pair oscillation period; dt=0.04 → T_pair/0.04 steps)');
console.log('  core_barrier = U(d=0.02) − U(d_min); positive = pairs need to climb this to overlap');
console.log();

// ────────────────────────────────────────────────────────────────────────────
// 4. Apply tuning to params and verify lattice equilibrium + stability.
// ────────────────────────────────────────────────────────────────────────────
const params = {
  ...DEFAULTS,
  config: 'head_body_tail',
  cutoff: 1.5,
  sphereR: 100,
  damping: 0.958,
  alignment_exponent: 2,
  bond_distance: BOND,
  bond_repulsion: 0,
  bond_attraction: -20,
  bond_alignment_torque: 20,
};
for (const s of solved) {
  const k = `hbt_${s.name.toLowerCase()}`;
  params[`${k}_distance`]   = s.dist;
  params[`${k}_repulsion`]  = s.rep;
  params[`${k}_attraction`] = s.att;
}

function hexPatch(rings) {
  const a = INTERMOLECULE_DIST;
  const a1 = [a, 0];
  const a2 = [a / 2, a * Math.sqrt(3) / 2];
  const positions = [];
  for (let i = -rings; i <= rings; i++)
    for (let j = -rings; j <= rings; j++) {
      const x = i * a1[0] + j * a2[0];
      const y = i * a1[1] + j * a2[1];
      if (Math.hypot(x, y) <= rings * a + 1e-9) positions.push([x, y]);
    }
  return positions;
}
function buildChains(positions) {
  const blocks = [];
  for (const [x, y] of positions) {
    const h = makeBlock([x, y,  0],          [0, 0, 0, 1]); h.type = 0;
    const b = makeBlock([x, y, -BOND],       [0, 0, 0, 1]); b.type = 1;
    const t = makeBlock([x, y, -2 * BOND],   [0, 0, 0, 1]); t.type = 2;
    h.bond_inner = b; b.bond_outer = h;
    b.bond_inner = t; t.bond_outer = b;
    blocks.push(h, b, t);
  }
  return blocks;
}
function computePairForces(blocks) {
  for (const b of blocks) {
    b.force = [0, 0, 0]; b.torque = [0, 0, 0];
    refreshNormal(b);
  }
  for (let i = 0; i < blocks.length; i++)
    for (let j = i + 1; j < blocks.length; j++)
      applyPair(blocks[i], blocks[j], params);
}

const positions = hexPatch(2);
const blocks = buildChains(positions);
const centerIdx = positions.findIndex(([x, y]) => Math.hypot(x, y) < 1e-9);
const centerH = blocks[3 * centerIdx + 0];
const centerB = blocks[3 * centerIdx + 1];
const centerT = blocks[3 * centerIdx + 2];
function fmag(b) { return Math.hypot(b.force[0], b.force[1], b.force[2]); }

computePairForces(blocks);
console.log('── Equilibrium check on a 19-chain hex patch ──');
console.log(`  |F_H_center| = ${fmag(centerH).toExponential(3)}`);
console.log(`  |F_B_center| = ${fmag(centerB).toExponential(3)}`);
console.log(`  |F_T_center| = ${fmag(centerT).toExponential(3)}`);
console.log();

console.log('── Lateral & vertical stability (per-chain spring constant) ──');
const DELTA = 0.005;
for (const [name, dx, dy, dz] of [['+x',DELTA,0,0],['+y',0,DELTA,0],['+z',0,0,DELTA]]) {
  for (let i = 0; i < positions.length; i++) {
    const [x, y] = positions[i];
    blocks[3*i+0].pos = [x, y, 0];
    blocks[3*i+1].pos = [x, y, -BOND];
    blocks[3*i+2].pos = [x, y, -2*BOND];
  }
  for (const b of [centerH, centerB, centerT]) { b.pos[0] += dx; b.pos[1] += dy; b.pos[2] += dz; }
  computePairForces(blocks);
  const fx = centerH.force[0] + centerB.force[0] + centerT.force[0];
  const fy = centerH.force[1] + centerB.force[1] + centerT.force[1];
  const fz = centerH.force[2] + centerB.force[2] + centerT.force[2];
  const k_chain = -(fx*dx + fy*dy + fz*dz) / (DELTA * DELTA);
  const T = k_chain > 0 ? 2 * Math.PI / Math.sqrt(k_chain / 3) : Infinity;
  console.log(`  perturb ${name} by ${DELTA}: k_chain = ${k_chain.toFixed(2)}, T_chain ≈ ${T.toFixed(2)}  (${k_chain > 0 ? 'restoring ✓' : 'NOT restoring ✗'})`);
}
console.log();

console.log('── Suggested DEFAULTS ──');
console.log(`  bond_distance: ${BOND}`);
console.log(`  alignment_exponent: ${params.alignment_exponent}`);
for (const s of solved) {
  const k = s.name.toLowerCase();
  console.log(`  hbt_${k}: dist=${s.dist.toFixed(5)}, rep=${s.rep}, att=${s.att}`);
}
