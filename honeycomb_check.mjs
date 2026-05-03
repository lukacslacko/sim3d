#!/usr/bin/env node
// Honeycomb / hex-layer analysis for HBT chains.
//
// We build a small 2D hex patch of 3-block chains, all standing perpendicular
// to the layer (H at top, B middle, T bottom), with the geometric NN distances
// derived below. Then we:
//   (1) verify pair forces are ~zero at the lattice (equilibrium check),
//   (2) perturb the central chain laterally and along z, measure restoring,
//   (3) report the radial spring constants and oscillation periods,
//   (4) print the chosen DEFAULTS so we can paste them in.

import {
  DEFAULTS, makeBlock, refreshNormal, applyPair, randomQuat,
} from './physics.mjs';

// ────────────────────────────────────────────────────────────────────────────
// 1. Geometry: molecule length 0.3, intermolecule spacing 0.3.
// ────────────────────────────────────────────────────────────────────────────
const MOLECULE_LENGTH    = 0.3;       // H to T span of one chain
const INTERMOLECULE_DIST = 0.3;       // NN center-to-center in the hex layer
const BOND               = MOLECULE_LENGTH / 2;     // 0.15 — H↔B and B↔T spacing

// In-layer NN: the two heads (or two bodies, two tails) sit at the same z, so
// the pair distance equals the in-layer spacing.
const D_HH = INTERMOLECULE_DIST;
const D_BB = INTERMOLECULE_DIST;
const D_TT = INTERMOLECULE_DIST;
// Cross-layer NN: a head at z=0 looks across to the body of a neighboring
// chain, which sits at (Δx, Δy, −BOND) with √(Δx²+Δy²) = INTERMOLECULE_DIST.
// Distance = √(INTER² + BOND²).
const D_HB = Math.hypot(INTERMOLECULE_DIST, BOND);
const D_BT = Math.hypot(INTERMOLECULE_DIST, BOND);
// Intra-chain HT (head and tail of the same molecule) = molecule length.
const D_HT = MOLECULE_LENGTH;

console.log('── Geometric NN distances for hex layer ──');
console.log(`  bond_distance    = ${BOND.toFixed(5)}    (chain length = ${MOLECULE_LENGTH})`);
console.log(`  HH / BB / TT     = ${D_HH.toFixed(5)}    (in-layer NN)`);
console.log(`  HB / BT          = ${D_HB.toFixed(5)}    (= √(${INTERMOLECULE_DIST}² + ${BOND}²))`);
console.log(`  HT (intra-chain) = ${D_HT.toFixed(5)}`);
console.log();

// ────────────────────────────────────────────────────────────────────────────
// 2. Pick attraction strengths.
//
// With rep=0, the pair energy is U = cf·att·W where W = 1/(1+(d−dist)²) and
// cf = ((1+n_a·n_b)/2)^k. At d=dist with normals parallel (cf=1):
//   d²U/dd² = -2·att      (positive curvature when att<0 → real well)
// so the radial spring constant is k_radial = -2·att. With dt=0.04 (default
// timestep) and damping=0.958, k_radial up to ~50 keeps each pair's natural
// period (~ 2π/√k) well above several timesteps. Choose:
//   |att| ≈ 1 for in-layer same-type (the structural skeleton)
//   |att| ≈ 0.5 for cross-layer cross-type (secondary cohesion)
// to keep timescales gentle and let the bond force (att=-20, k≈40) dominate
// chain integrity.
//
// alignment_exponent: at the equilibrium all chains are aligned (cf=1 for any
// k), so the exponent doesn't change forces at the lattice — but it *does*
// govern how sharply attraction cuts off when a chain tilts. k=2 keeps a
// gentle restoring envelope and helps convergence from random initial
// orientations (k=5 is too sharp — disaligned chains feel almost no
// attraction and converge slowly).
// ────────────────────────────────────────────────────────────────────────────
const params = {
  ...DEFAULTS,
  config: 'head_body_tail',
  cutoff: 1.5,
  sphereR: 100,                    // big — keep walls out of the way
  damping: 0.958,
  alignment_exponent: 2,
  bond_distance: BOND,
  bond_repulsion: 0,
  bond_attraction: -20,
  bond_alignment_torque: 20,
  hbt_hh_distance: D_HH, hbt_hh_repulsion: 0, hbt_hh_attraction: -1.0,
  hbt_bb_distance: D_BB, hbt_bb_repulsion: 0, hbt_bb_attraction: -1.0,
  hbt_tt_distance: D_TT, hbt_tt_repulsion: 0, hbt_tt_attraction: -1.0,
  // Cross-layer attractions zeroed: the bond already provides cohesion across
  // the H/B/T layers (att=-20, k≈40 — vastly stiffer than any pair force we'd
  // pick). Including HB/BT introduces a chain-compression bias from the
  // 2nd-NN tail of the attractive well (each H sees 6 cross-chain Bs at
  // z=-BOND but no symmetric partners above), which we'd then have to fight.
  // With HB=BT=0 the chains stand straight at exactly 2·bond_distance.
  hbt_hb_distance: D_HB, hbt_hb_repulsion: 0, hbt_hb_attraction: 0,
  hbt_ht_distance: D_HT, hbt_ht_repulsion: 0, hbt_ht_attraction: 0,
  hbt_bt_distance: D_BT, hbt_bt_repulsion: 0, hbt_bt_attraction: 0,
};

console.log('── Pair spring constants k_radial = −2·cf·att (cf=1 in the lattice) ──');
for (const [name, att] of [
  ['HH', params.hbt_hh_attraction], ['BB', params.hbt_bb_attraction],
  ['TT', params.hbt_tt_attraction], ['HB', params.hbt_hb_attraction],
  ['BT', params.hbt_bt_attraction],
]) {
  const k = -2 * att;
  const T = 2 * Math.PI / Math.sqrt(Math.max(k, 1e-9));
  console.log(`  ${name}: k = ${k.toFixed(3)}, period T = ${T.toFixed(2)} time units (${(T/0.04).toFixed(0)} steps @ dt=0.04)`);
}
console.log();

// ────────────────────────────────────────────────────────────────────────────
// 3. Build a 19-chain hex patch (1 center + 6 NN + 12 second-NN). Larger than
// the minimum 7-chain neighborhood so the central chain sees realistic
// surroundings beyond its NN shell.
// ────────────────────────────────────────────────────────────────────────────
function hexPatch(rings) {
  const a = INTERMOLECULE_DIST;
  const a1 = [a, 0];
  const a2 = [a / 2, a * Math.sqrt(3) / 2];
  const positions = [];
  for (let i = -rings; i <= rings; i++)
    for (let j = -rings; j <= rings; j++) {
      // Drop sites outside the hex with circumradius "rings".
      const x = i * a1[0] + j * a2[0];
      const y = i * a1[1] + j * a2[1];
      if (Math.hypot(x, y) <= rings * a + 1e-9) positions.push([x, y]);
    }
  return positions;
}

function buildChains(positions, p) {
  // Identity quaternion → normal = +Z; chain stacked along −Z so the bond
  // direction (inner → outer = tail → head) is also +Z, which is the stable
  // direction for bond_alignment_torque.
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

// Apply ONLY pair forces (no bonds — the chains are at exact bond equilibrium
// so bond contributions are zero anyway, but skipping avoids depending on the
// non-exported applyBondForce).
function computePairForces(blocks, p) {
  for (const b of blocks) {
    b.force = [0, 0, 0]; b.torque = [0, 0, 0];
    refreshNormal(b);
  }
  for (let i = 0; i < blocks.length; i++)
    for (let j = i + 1; j < blocks.length; j++)
      applyPair(blocks[i], blocks[j], params);
}

// ────────────────────────────────────────────────────────────────────────────
// 4. Equilibrium check: forces on the CENTER chain should be ~0. (Edge chains
// will have residual force pointing inward — that's the surface tension of
// our finite patch and not a problem for an extended/closed lattice.)
// ────────────────────────────────────────────────────────────────────────────
const positions = hexPatch(2);
const blocks = buildChains(positions, params);
const centerIdx = positions.findIndex(([x, y]) => Math.hypot(x, y) < 1e-9);
const centerH = blocks[3 * centerIdx + 0];
const centerB = blocks[3 * centerIdx + 1];
const centerT = blocks[3 * centerIdx + 2];

computePairForces(blocks, params);
function fmag(b) { return Math.hypot(b.force[0], b.force[1], b.force[2]); }
function tmag(b) { return Math.hypot(b.torque[0], b.torque[1], b.torque[2]); }

console.log('── Equilibrium check (center chain) ──');
console.log(`  patch size: ${positions.length} chains (rings=2)`);
console.log(`  |F_H| = ${fmag(centerH).toExponential(3)}    |τ_H| = ${tmag(centerH).toExponential(3)}`);
console.log(`  |F_B| = ${fmag(centerB).toExponential(3)}    |τ_B| = ${tmag(centerB).toExponential(3)}`);
console.log(`  |F_T| = ${fmag(centerT).toExponential(3)}    |τ_T| = ${tmag(centerT).toExponential(3)}`);
console.log();

// ────────────────────────────────────────────────────────────────────────────
// 5. Stability: perturb the entire central chain by δ in each axis, recompute
// forces on its three blocks, sum to a "chain force", and verify it's
// restoring (anti-parallel to δ). The slope F·(−δ̂)/|δ| is the per-chain
// effective spring constant in that direction.
// ────────────────────────────────────────────────────────────────────────────
console.log('── Lateral & vertical stability of the central chain ──');
const DELTA = 0.005;
for (const [axisName, dx, dy, dz] of [
  ['+x', DELTA, 0, 0],
  ['+y', 0, DELTA, 0],
  ['+z', 0, 0, DELTA],
]) {
  // Reset to lattice positions
  for (let i = 0; i < positions.length; i++) {
    const [x, y] = positions[i];
    blocks[3*i+0].pos = [x, y, 0];
    blocks[3*i+1].pos = [x, y, -BOND];
    blocks[3*i+2].pos = [x, y, -2*BOND];
  }
  // Displace center chain
  for (const b of [centerH, centerB, centerT]) {
    b.pos[0] += dx; b.pos[1] += dy; b.pos[2] += dz;
  }
  computePairForces(blocks, params);
  // Sum chain force
  const fx = centerH.force[0] + centerB.force[0] + centerT.force[0];
  const fy = centerH.force[1] + centerB.force[1] + centerT.force[1];
  const fz = centerH.force[2] + centerB.force[2] + centerT.force[2];
  // Component along −displacement (positive = restoring).
  const dotMinusDisp = -(fx * dx + fy * dy + fz * dz) / DELTA;
  const k_chain = dotMinusDisp / DELTA;   // effective spring constant
  const restoringMsg = k_chain > 0 ? 'restoring ✓' : 'NOT restoring ✗';
  const T = k_chain > 0 ? 2 * Math.PI / Math.sqrt(k_chain / 3) : Infinity;  // 3 blocks per chain
  console.log(`  perturb ${axisName} by ${DELTA}: F_chain = (${fx.toFixed(3)}, ${fy.toFixed(3)}, ${fz.toFixed(3)})`);
  console.log(`    k_chain = ${k_chain.toFixed(3)}  (${restoringMsg}, period T ≈ ${T === Infinity ? '∞' : T.toFixed(2)})`);
}
console.log();

// ────────────────────────────────────────────────────────────────────────────
// 6. Tilt stability: rotate the central chain's orientation slightly and
// check that the bond_alignment_torque + (1+cos)^k attraction restores.
// Here we only check the (1+cos)^k contribution (since pair forces alone are
// what we control here; the bond torque trivially restores tilt).
// ────────────────────────────────────────────────────────────────────────────
console.log('── Tilt stability of the central chain (pair-force contribution only) ──');
const TILT = 0.05;  // ~3°
for (const axisName of ['x', 'y']) {
  // Reset positions
  for (let i = 0; i < positions.length; i++) {
    const [x, y] = positions[i];
    blocks[3*i+0].pos = [x, y, 0];
    blocks[3*i+1].pos = [x, y, -BOND];
    blocks[3*i+2].pos = [x, y, -2*BOND];
    blocks[3*i+0].quat = [0, 0, 0, 1];
    blocks[3*i+1].quat = [0, 0, 0, 1];
    blocks[3*i+2].quat = [0, 0, 0, 1];
  }
  // Rotate central chain about world-x or world-y axis by TILT.
  // q = [sin(θ/2)·n, cos(θ/2)] with n along axis.
  const c = Math.cos(TILT / 2), s = Math.sin(TILT / 2);
  const q = axisName === 'x' ? [s, 0, 0, c] : [0, s, 0, c];
  for (const b of [centerH, centerB, centerT]) b.quat = [...q];
  computePairForces(blocks, params);
  // Total torque on center chain
  const tx = centerH.torque[0] + centerB.torque[0] + centerT.torque[0];
  const ty = centerH.torque[1] + centerB.torque[1] + centerT.torque[1];
  const tz = centerH.torque[2] + centerB.torque[2] + centerT.torque[2];
  // Restoring torque is ABOUT (-axis): if rotated about +x, restoring torque is along -x.
  const rest = axisName === 'x' ? -tx : -ty;
  console.log(`  tilt ${axisName} by ${TILT} rad: τ_chain = (${tx.toFixed(4)}, ${ty.toFixed(4)}, ${tz.toFixed(4)})`);
  console.log(`    restoring torque component = ${rest.toExponential(3)}  ${rest > 0 ? 'restoring ✓' : 'destabilizing ✗ (relies on bond_alignment_torque)'}`);
}
console.log();

// ────────────────────────────────────────────────────────────────────────────
// 7. Print the final DEFAULTS-style block.
// ────────────────────────────────────────────────────────────────────────────
console.log('── Suggested DEFAULTS overrides ──');
console.log(`  bond_distance: ${BOND}`);
console.log(`  alignment_exponent: ${params.alignment_exponent}`);
console.log(`  hbt_hh: dist=${D_HH.toFixed(5)}, rep=0, att=${params.hbt_hh_attraction}`);
console.log(`  hbt_bb: dist=${D_BB.toFixed(5)}, rep=0, att=${params.hbt_bb_attraction}`);
console.log(`  hbt_tt: dist=${D_TT.toFixed(5)}, rep=0, att=${params.hbt_tt_attraction}`);
console.log(`  hbt_hb: dist=${D_HB.toFixed(5)}, rep=0, att=${params.hbt_hb_attraction}`);
console.log(`  hbt_ht: dist=${D_HT.toFixed(5)}, rep=0, att=${params.hbt_ht_attraction} (intra-chain only)`);
console.log(`  hbt_bt: dist=${D_BT.toFixed(5)}, rep=0, att=${params.hbt_bt_attraction}`);
