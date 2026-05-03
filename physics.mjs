// Shared physics for the sim3d playground.
// Pure JS — no THREE dependency, so this also runs under node for analysis.
//
// All vectors are length-3 number[] (mutable). Quaternions are [x,y,z,w].
// A block has its sheet normal along its local +Z axis.

export const DEFAULTS = {
  // Active block-type configuration. Selects which pair model + initial
  // typing/connectivity is used. See applyPair() for dispatch.
  //   'sheet'           — single-type model with orientation-rich affinity
  //   'head_tail'       — head bonded to exactly one tail (chain length 2)
  //   'head_body_tail'  — three types in a chain head←body→tail (chain
  //                       length 3); each of the 6 unordered unbonded pair
  //                       types (HH, BB, TT, HB, HT, BT) has its own
  //                       distance/repulsion/attraction trio (hbt_*_* below).
  //                       Bonds (head↔body, body↔tail) use the same shared
  //                       bond_* + bond_alignment_torque as head_tail.
  config: 'head_body_tail',

  // ── shared parameters (apply to every config) ─────────────────────────
  sphereR: 9,
  damping: 0.989,
  wallK: 0.1,
  forceCap: 50,
  // Interaction cutoff: pair energies are multiplied by a smooth mask m(d)
  // that is 1 for d ≤ cutoff/2 and tapers to 0 at d = cutoff. The spatial
  // hash in stepBlocks then skips any pair whose cells aren't adjacent —
  // exact, since pairs past cutoff contribute zero. Bonded pairs in the
  // head_tail config are exempt: they're force-applied unconditionally so
  // the bond never gets "lost" by separation.
  cutoff: 3.0,

  // ── 'sheet' config parameters ─────────────────────────────────────────
  // Sign convention: -1/d times a NEGATIVE param yields a true positive
  // repulsion at small d (since -1/d × negative = +positive/d). Likewise
  // a negative `attraction` makes the well at d_peak an energy minimum.
  in_plane_repulsion:   -0.6,
  orthogonal_repulsion: -1.2,
  attraction:           -1.2,
  d_peak:                0.4,
  bend:                  0.7,

  // ── 'head_tail' config parameters ─────────────────────────────────────
  // Sign convention for *_repulsion / *_attraction: negative numbers give
  // physical repulsion + attractive well in the "want distance a" form
  //   U = −repulsion/d + attraction / (1 + (d − a)²).
  // (−rep/d × negative = +positive/d → real repulsion at small d.
  //  attraction × negative = energy minimum at d = a.)

  // Bonded H–T pair: no (1+cos θ)/2 factor (orientation is set by the
  // bond_alignment_torque below).
  bond_distance:                     0.4,
  bond_repulsion:                    0.0,
  bond_attraction:                 -20.0,
  // Head–head pair (always unbonded). Attraction multiplied by
  // (1 + n_a·n_b)/2 — parallel heads fully attract, antiparallel ones repel.
  head_head_distance:                0.9,
  head_head_repulsion:              -0.1,
  head_head_attraction:              0.0,
  // Tail–tail pair (always unbonded). Same (1+cos θ)/2 factor as H–H.
  tail_tail_distance:                0.5,
  tail_tail_repulsion:               0.0,
  tail_tail_attraction:             -0.15,
  // Unbonded H–T pair (a head and a tail that aren't bonded to each other).
  // Pure −k/d affinity, no equilibrium — they just pull each other in until
  // something else (H–H or T–T repulsion) stops them.
  unbonded_ht_attraction:            0.0,
  // Each bonded block feels a torque τ = strength · (n × bond_dir) pulling
  // its direction parallel to the bond axis (tail → head). Antiparallel is
  // the unstable equilibrium; parallel is the stable one.
  bond_alignment_torque:            20.0,

  // ── 'head_body_tail' config parameters ────────────────────────────────
  // Six unbonded pair types (3 same-type + 3 cross-type). All use the
  // shared "want distance" form
  //   U = -repulsion/d + (1 + n_a·n_b)/2 · attraction / (1 + (d − distance)²)
  // with the same sign convention as the rest of this file (negative
  // repulsion = real core repulsion; negative attraction = attractive well
  // at `distance`). Bonded H–B and B–T pairs are skipped here and handled
  // by applyBondForce instead. Defaults below preserve the prior collapsed
  // behavior (BB-only attraction; HH soft repulsion; everything else off).
  hbt_hh_distance: 0.9, hbt_hh_repulsion: -0.1, hbt_hh_attraction:  0.0,
  hbt_bb_distance: 0.5, hbt_bb_repulsion:  0.0, hbt_bb_attraction: -0.15,
  hbt_tt_distance: 0.9, hbt_tt_repulsion: -0.1, hbt_tt_attraction:  0.0,
  hbt_hb_distance: 0.5, hbt_hb_repulsion:  0.0, hbt_hb_attraction:  0.0,
  hbt_ht_distance: 0.9, hbt_ht_repulsion:  0.0, hbt_ht_attraction:  0.0,
  hbt_bt_distance: 0.5, hbt_bt_repulsion:  0.0, hbt_bt_attraction:  0.0,
};

export function makeBlock(pos, quat, opts = {}) {
  return {
    pos:    [pos[0], pos[1], pos[2]],
    quat:   [quat[0], quat[1], quat[2], quat[3]],
    vel:    [0, 0, 0],
    ang:    [0, 0, 0],
    force:  [0, 0, 0],
    torque: [0, 0, 0],
    normal: [0, 0, 0],
    // Type ids per config:
    //   head_tail:       0 = head, 1 = tail
    //   head_body_tail:  0 = head, 1 = body, 2 = tail
    type:       opts.type       ?? 0,
    // Two bond slots, ordered so each bond is visited exactly once when
    // iterating .bond_inner. For an unbonded block both are null.
    //   bond_inner: the partner this block is the "outer" of
    //   bond_outer: the partner this block is the "inner" of
    // Chain head←body→tail in HBT:
    //   head.bond_inner = body
    //   body.bond_outer = head ; body.bond_inner = tail
    //   tail.bond_outer = body
    bond_inner: opts.bond_inner ?? null,
    bond_outer: opts.bond_outer ?? null,
  };
}

// Set up head_tail typing/connectivity. First half → heads, second → tails;
// head[i] bonded to tail[i]. Tails are nudged near their bonded head so the
// bond force starts near equilibrium.
export function setupHeadTail(blocks, p) {
  const N = blocks.length;
  const half = N >> 1;
  for (let i = 0; i < N; i++) {
    blocks[i].type = i < half ? 0 : 1;
    blocks[i].bond_inner = null;
    blocks[i].bond_outer = null;
  }
  const off = (p.bond_distance ?? 0.5);
  for (let i = 0; i < half; i++) {
    const head = blocks[i];
    const tail = blocks[half + i];
    head.bond_inner = tail;
    tail.bond_outer = head;
    tail.pos[0] = head.pos[0] + (Math.random() * 2 - 1) * off;
    tail.pos[1] = head.pos[1] + (Math.random() * 2 - 1) * off;
    tail.pos[2] = head.pos[2] + (Math.random() * 2 - 1) * off;
  }
}

// Set up head_body_tail typing/connectivity. First third → heads, middle
// third → bodies, last third → tails. head[i]–body[i]–tail[i] forms a
// 3-block chain bonded twice. Body and tail are placed on a random axis
// from the head at spacings ≈ bond_distance so each chain starts near
// straight at near-equilibrium spacing.
export function setupHeadBodyTail(blocks, p) {
  const N = blocks.length;
  const third = (N / 3) | 0;
  for (let i = 0; i < N; i++) {
    blocks[i].type = i < third ? 0 : (i < 2 * third ? 1 : 2);
    blocks[i].bond_inner = null;
    blocks[i].bond_outer = null;
  }
  const off = (p.bond_distance ?? 0.4);
  for (let i = 0; i < third; i++) {
    const head = blocks[i];
    const body = blocks[third + i];
    const tail = blocks[2 * third + i];
    head.bond_inner = body;
    body.bond_outer = head;
    body.bond_inner = tail;
    tail.bond_outer = body;

    // Random unit axis; body sits 1·off back from head, tail another 1·off.
    let dx = Math.random() * 2 - 1;
    let dy = Math.random() * 2 - 1;
    let dz = Math.random() * 2 - 1;
    const norm = Math.hypot(dx, dy, dz) || 1;
    dx /= norm; dy /= norm; dz /= norm;
    body.pos[0] = head.pos[0] - dx * off;
    body.pos[1] = head.pos[1] - dy * off;
    body.pos[2] = head.pos[2] - dz * off;
    tail.pos[0] = body.pos[0] - dx * off;
    tail.pos[1] = body.pos[1] - dy * off;
    tail.pos[2] = body.pos[2] - dz * off;
  }
}

// Rotate v by quaternion q.
export function quatRotate(q, v, out = [0,0,0]) {
  const [x, y, z, w] = q;
  const [vx, vy, vz] = v;
  // t = 2 * (q.xyz × v)
  const tx = 2 * (y * vz - z * vy);
  const ty = 2 * (z * vx - x * vz);
  const tz = 2 * (x * vy - y * vx);
  // v' = v + w*t + q.xyz × t
  out[0] = vx + w * tx + (y * tz - z * ty);
  out[1] = vy + w * ty + (z * tx - x * tz);
  out[2] = vz + w * tz + (x * ty - y * tx);
  return out;
}

export function refreshNormal(b) {
  quatRotate(b.quat, Z_HAT, b.normal);
}
const Z_HAT = [0, 0, 1];

// ---------------------------------------------------------------------------
// Pair affinity model.
//
// User-defined form (with θ = angle between the line of centers and each
// block's sheet plane, so θ = 0 → in-plane displacement, θ = π/2 → stacked):
//
//   U = (-1/d) · (P_in + (P_ortho − P_in) · sin θ)
//       + cos θ · A / (1 + (d − d_peak)²)
//
// where P_in = in_plane_repulsion, P_ortho = orthogonal_repulsion, A = attraction.
//
// Since cA = rhat · n_a is the projection on the normal — i.e. sin θ — we use
// per-block squared averages (smoother derivatives than |sin|/|cos|, and the
// limits match exactly):
//
//   T = ½ (cA² + cB²)     ↔ averaged sin²θ   (stackedness)
//   S = 1 − T             ↔ averaged cos²θ   (in-planeness)
//
//   U = (-1/d) · (P_in + (P_ortho − P_in) · T)  +  S · A · W,
//   W = 1 / (1 + (d − d_peak)²).
//
// Limits (with the default negative parameters making these actual
// repulsions / attractions in energy):
//   in-plane (θ = 0,  T = 0): U = −P_in/d  + A·W   = +|P_in|/d − |A|·W
//                              → core repulsion + attractive well at d_peak.
//   stacked  (θ = π/2, T = 1): U = −P_ortho/d      = +|P_ortho|/d
//                              → pure repulsion (blocks face-to-face push off).
//
// The angular interpolation is linear in T (≡ sin²θ); the original prompt
// wrote it as linear in |sin θ|. The two agree at the two limits where the
// parameter names are anchored; the squared form keeps derivatives smooth
// near stacked / in-plane (avoiding the sqrt-and-abs spikes you'd otherwise
// get and which would destabilize the integrator).
// ---------------------------------------------------------------------------

// Smooth cutoff mask. m(d) = 1 for d ≤ R/2, smoothstep-tapered to 0 at d = R.
//   t = (d − R/2) / (R/2)        ∈ [0, 1] in the taper region
//   m = 1 − (3 t² − 2 t³)
//   dm/dd = (−6 t + 6 t²) · (2 / R)
// The taper is C¹ (continuous value and first derivative) at both endpoints,
// so forces and torques don't kink at d = R/2 or d = R.
function smoothMask(d, cutoff) {
  const half = 0.5 * cutoff;
  if (d <= half) return 1;
  if (d >= cutoff) return 0;
  const t = (d - half) / half;
  return 1 - (3 - 2 * t) * t * t;
}
function smoothMaskDeriv(d, cutoff) {
  const half = 0.5 * cutoff;
  if (d <= half || d >= cutoff) return 0;
  const t = (d - half) / half;
  return (6 * t * t - 6 * t) / half;
}

// Bent angle factors. Returns [T, dT/dcA, dT/dcB] where
//   T = ½ ( sin²(θ_a − β) + sin²(θ_b − β) )
// with sin(θ_a) = cA, sin(θ_b) = −cB (the displacement points the opposite
// way as seen from b), cos(θ_*) = √(1 − c*²) ≥ 0, and β = p.bend.
//
// Floor on (1 − c²) keeps the √'s derivative finite if a transient kick
// drives a block to nearly-stacked. The equilibrium with bend > 0 sits
// well away from |c| = 1, so this floor is only ever exercised in flight.
function _bentT(cA, cB, p) {
  const beta = p.bend;
  if (beta === 0) {
    const T = 0.5 * (cA*cA + cB*cB);
    return [T, cA, cB];
  }
  const cb = Math.cos(beta), sb = Math.sin(beta);
  const sq2A = 1 - cA*cA, sq2B = 1 - cB*cB;
  const sqA = Math.sqrt(sq2A < 1e-8 ? 1e-8 : sq2A);
  const sqB = Math.sqrt(sq2B < 1e-8 ? 1e-8 : sq2B);
  const sA =  cA*cb - sqA*sb;   // sin(θ_a − β)
  const sB = -cB*cb - sqB*sb;   // sin(θ_b − β),  θ_b uses −cB
  const T = 0.5 * (sA*sA + sB*sB);
  const dsA_dcA =  cb + (cA / sqA) * sb;       // d/dcA [cA cb − √(1−cA²) sb]
  const dsB_dcB = -cb + (cB / sqB) * sb;       // d/dcB [−cB cb − √(1−cB²) sb]
  return [T, sA * dsA_dcA, sB * dsB_dcB];
}

// Unmasked pair energy U₀(d, cA, cB). The cutoff multiplies this in
// pairAffinity / pairAffinityGrad; isolated for clarity and reuse.
function pairAffinityRaw(d, cA, cB, p) {
  const [T] = _bentT(cA, cB, p);
  const S = 1 - T;
  const dpk = d - p.d_peak;
  const W = 1 / (1 + dpk * dpk);
  const pre = p.in_plane_repulsion + (p.orthogonal_repulsion - p.in_plane_repulsion) * T;
  return -pre / d + S * p.attraction * W;
}
function pairAffinityRawGrad(d, cA, cB, p) {
  const [T, dT_dcA, dT_dcB] = _bentT(cA, cB, p);
  const S = 1 - T;
  const dpk = d - p.d_peak;
  const W = 1 / (1 + dpk * dpk);
  const dW_dd = -2 * dpk * W * W;
  const pre = p.in_plane_repulsion + (p.orthogonal_repulsion - p.in_plane_repulsion) * T;
  const dU_dd  = pre / (d * d) + S * p.attraction * dW_dd;
  const dU_dT  = -(p.orthogonal_repulsion - p.in_plane_repulsion) / d - p.attraction * W;
  return [dU_dd, dU_dT * dT_dcA, dU_dT * dT_dcB];
}

export function pairAffinity(d, cA, cB, p) {
  const m = smoothMask(d, p.cutoff);
  if (m === 0) return 0;
  return m * pairAffinityRaw(d, cA, cB, p);
}

// Returns [∂(m·U₀)/∂d, ∂(m·U₀)/∂cA, ∂(m·U₀)/∂cB].
// m depends only on d, so cA/cB partials just pick up a factor of m.
//   ∂U/∂d   = m · ∂U₀/∂d  +  U₀ · dm/dd
export function pairAffinityGrad(d, cA, cB, p) {
  const m = smoothMask(d, p.cutoff);
  if (m === 0) return [0, 0, 0];
  const [dU_dd0, dU_dcA0, dU_dcB0] = pairAffinityRawGrad(d, cA, cB, p);
  const dm = smoothMaskDeriv(d, p.cutoff);
  if (dm === 0) return [m * dU_dd0, m * dU_dcA0, m * dU_dcB0];
  const U0 = pairAffinityRaw(d, cA, cB, p);
  return [m * dU_dd0 + dm * U0, m * dU_dcA0, m * dU_dcB0];
}

// Pair force/torque dispatch — selects the per-config implementation.
export function applyPair(a, b, p) {
  if (p.config === 'head_body_tail') return applyPairHeadBodyTail(a, b, p);
  if (p.config === 'head_tail')      return applyPairHeadTail(a, b, p);
  return applyPairSheet(a, b, p);
}

// Accumulate pair force on a, b and torques (about each block's own normal).
const _r = [0,0,0], _rhat = [0,0,0], _fb = [0,0,0];
function applyPairSheet(a, b, p) {
  _r[0] = b.pos[0] - a.pos[0];
  _r[1] = b.pos[1] - a.pos[1];
  _r[2] = b.pos[2] - a.pos[2];
  // Cheap reject before the sqrt: d² > cutoff² ⇒ mask is zero anyway.
  const d2 = _r[0]*_r[0] + _r[1]*_r[1] + _r[2]*_r[2];
  if (d2 > p.cutoff * p.cutoff) return 0;
  let d = Math.sqrt(d2);
  if (d < 1e-4) {
    // Degenerate overlap — kick apart along a random axis to break symmetry.
    const k = 5;
    const rx = Math.random()-0.5, ry = Math.random()-0.5, rz = Math.random()-0.5;
    a.force[0] -= rx*k; a.force[1] -= ry*k; a.force[2] -= rz*k;
    b.force[0] += rx*k; b.force[1] += ry*k; b.force[2] += rz*k;
    return 1;
  }
  const inv_d = 1 / d;
  _rhat[0] = _r[0] * inv_d; _rhat[1] = _r[1] * inv_d; _rhat[2] = _r[2] * inv_d;

  const nA = a.normal, nB = b.normal;
  const cA = _rhat[0]*nA[0] + _rhat[1]*nA[1] + _rhat[2]*nA[2];
  const cB = _rhat[0]*nB[0] + _rhat[1]*nB[1] + _rhat[2]*nB[2];

  const [dU_dd, dU_dcA, dU_dcB] = pairAffinityGrad(d, cA, cB, p);

  // Force on b = -∂U/∂d · rhat - ∂U/∂cA · (nA - cA·rhat)/d - ∂U/∂cB · (nB - cB·rhat)/d
  const ka = -dU_dcA * inv_d, kb = -dU_dcB * inv_d;
  _fb[0] = -dU_dd*_rhat[0] + ka*(nA[0] - cA*_rhat[0]) + kb*(nB[0] - cB*_rhat[0]);
  _fb[1] = -dU_dd*_rhat[1] + ka*(nA[1] - cA*_rhat[1]) + kb*(nB[1] - cB*_rhat[1]);
  _fb[2] = -dU_dd*_rhat[2] + ka*(nA[2] - cA*_rhat[2]) + kb*(nB[2] - cB*_rhat[2]);

  a.force[0] -= _fb[0]; a.force[1] -= _fb[1]; a.force[2] -= _fb[2];
  b.force[0] += _fb[0]; b.force[1] += _fb[1]; b.force[2] += _fb[2];

  // Torque on a from this pair: -nA × ∂U/∂nA = -dU_dcA · (nA × rhat)
  // (and same form for b)
  const tax = nA[1]*_rhat[2] - nA[2]*_rhat[1];
  const tay = nA[2]*_rhat[0] - nA[0]*_rhat[2];
  const taz = nA[0]*_rhat[1] - nA[1]*_rhat[0];
  a.torque[0] -= dU_dcA * tax;
  a.torque[1] -= dU_dcA * tay;
  a.torque[2] -= dU_dcA * taz;

  const tbx = nB[1]*_rhat[2] - nB[2]*_rhat[1];
  const tby = nB[2]*_rhat[0] - nB[0]*_rhat[2];
  const tbz = nB[0]*_rhat[1] - nB[1]*_rhat[0];
  b.torque[0] -= dU_dcB * tbx;
  b.torque[1] -= dU_dcB * tby;
  b.torque[2] -= dU_dcB * tbz;
  return 1;
}

// ---------------------------------------------------------------------------
// head_tail config — pair forces and bond force/torque.
// ---------------------------------------------------------------------------

// Bond force/torque: applied unconditionally (no cutoff) so a bond never
// gets "lost" by separation. Pure radial "want distance bond_distance" force
// (-bond_repulsion/d + bond_attraction·W), plus an alignment torque pulling
// each block's direction toward bond_dir = (outer.pos − inner.pos)/d.
//
// Used by both head_tail (one bond per chain) and head_body_tail (two bonds
// per chain — head←body and body←tail; both use the same shared bond_*
// parameters and both align block.direction with their bond_dir, which is
// consistent because the chain is colinear at equilibrium).
function applyBondForce(outer, inner, p) {
  const rx = outer.pos[0] - inner.pos[0];
  const ry = outer.pos[1] - inner.pos[1];
  const rz = outer.pos[2] - inner.pos[2];
  const d2 = rx*rx + ry*ry + rz*rz;
  if (d2 < 1e-8) {
    const k = 5;
    const px = Math.random()-0.5, py = Math.random()-0.5, pz = Math.random()-0.5;
    outer.force[0] += px*k; outer.force[1] += py*k; outer.force[2] += pz*k;
    inner.force[0] -= px*k; inner.force[1] -= py*k; inner.force[2] -= pz*k;
    return;
  }
  const d = Math.sqrt(d2);
  const inv_d = 1 / d;
  const bx = rx * inv_d, by = ry * inv_d, bz = rz * inv_d;  // bond_dir = inner → outer

  const da = d - p.bond_distance;
  const W  = 1 / (1 + da * da);
  const dU_dd = p.bond_repulsion / d2 - 2 * da * p.bond_attraction * W * W;
  const fmag = -dU_dd;
  outer.force[0] += fmag * bx;
  outer.force[1] += fmag * by;
  outer.force[2] += fmag * bz;
  inner.force[0] -= fmag * bx;
  inner.force[1] -= fmag * by;
  inner.force[2] -= fmag * bz;

  // Alignment torque on each block: τ = bond_alignment_torque · (n × bond_dir).
  // Parallel n with bond_dir is the stable equilibrium.
  const ak = p.bond_alignment_torque;
  if (ak !== 0) {
    const no = outer.normal;
    outer.torque[0] += ak * (no[1]*bz - no[2]*by);
    outer.torque[1] += ak * (no[2]*bx - no[0]*bz);
    outer.torque[2] += ak * (no[0]*by - no[1]*bx);
    const ni = inner.normal;
    inner.torque[0] += ak * (ni[1]*bz - ni[2]*by);
    inner.torque[1] += ak * (ni[2]*bx - ni[0]*bz);
    inner.torque[2] += ak * (ni[0]*by - ni[1]*bx);
  }
}

// Non-bonded pair in head_tail config. Cases:
//   • H–H, T–T (same type): "want distance head_head_distance / tail_tail_distance",
//     with the attraction term multiplied by (1 + n_a·n_b)/2 — parallel pairs
//     fully attract, antiparallel ones get no attraction (bare 1/d repulsion
//     wins, they repel). Force only; no torque.
//   • H–T not bonded: pure -unbonded_ht_attraction/d affinity (no equilibrium,
//     no cos factor, no torque).
// Bonded H–T pairs reach this function via the spatial hash too — we early-
// reject them so their force is only applied via applyBondForce (no double-
// counting).
function applyPairHeadTail(a, b, p) {
  // Bonded? Handled separately by applyBondForce in stepBlocks.
  if (a.bond_inner === b || a.bond_outer === b) return 0;

  const rx = b.pos[0] - a.pos[0];
  const ry = b.pos[1] - a.pos[1];
  const rz = b.pos[2] - a.pos[2];
  const d2 = rx*rx + ry*ry + rz*rz;
  if (d2 > p.cutoff * p.cutoff) return 0;
  if (d2 < 1e-8) {
    const k = 5;
    const px = Math.random()-0.5, py = Math.random()-0.5, pz = Math.random()-0.5;
    a.force[0] -= px*k; a.force[1] -= py*k; a.force[2] -= pz*k;
    b.force[0] += px*k; b.force[1] += py*k; b.force[2] += pz*k;
    return 1;
  }
  const d = Math.sqrt(d2);
  const inv_d = 1 / d;
  const m  = smoothMask(d, p.cutoff);
  if (m === 0) return 1;
  const dm = smoothMaskDeriv(d, p.cutoff);
  let U0, dU0_dd;
  if (a.type !== b.type) {
    // Unbonded H–T: U = -unbonded_ht_attraction / d.
    U0 = -p.unbonded_ht_attraction * inv_d;
    dU0_dd = p.unbonded_ht_attraction * inv_d * inv_d;   // d/dd[-k/d] = +k/d²
  } else {
    // Same type: H–H or T–T.
    const want_a   = a.type === 0 ? p.head_head_distance   : p.tail_tail_distance;
    const want_rep = a.type === 0 ? p.head_head_repulsion  : p.tail_tail_repulsion;
    const want_att = a.type === 0 ? p.head_head_attraction : p.tail_tail_attraction;
    const alpha = a.normal[0]*b.normal[0] + a.normal[1]*b.normal[1] + a.normal[2]*b.normal[2];
    const cf = (1 + alpha) * 0.5;        // (1 + cos θ) / 2
    const da = d - want_a;
    const W  = 1 / (1 + da * da);
    U0     = -want_rep * inv_d + cf * want_att * W;
    dU0_dd =  want_rep * inv_d * inv_d - 2 * da * cf * want_att * W * W;
  }

  const dU_dd = m * dU0_dd + dm * U0;
  const fmag = -dU_dd;
  const fx = fmag * rx * inv_d;
  const fy = fmag * ry * inv_d;
  const fz = fmag * rz * inv_d;
  a.force[0] -= fx; a.force[1] -= fy; a.force[2] -= fz;
  b.force[0] += fx; b.force[1] += fy; b.force[2] += fz;
  return 1;
}

// Non-bonded pair in head_body_tail config. Six unordered (a.type, b.type)
// combinations dispatch to their own want_distance / repulsion / attraction
// trio (hbt_hh_*, hbt_bb_*, hbt_tt_*, hbt_hb_*, hbt_ht_*, hbt_bt_*).
// All six use the same form:
//     U = −repulsion/d + (1 + n_a·n_b)/2 · attraction / (1 + (d − distance)²)
// Bonded pairs (head↔body or body↔tail in the same chain) are skipped here
// and handled unconditionally by applyBondForce.
//
// Type ids: 0 = H, 1 = B, 2 = T. Encoding the unordered pair as
// (min·3 + max) gives keys 0,1,2,4,5,8 for HH,HB,HT,BB,BT,TT — a flat
// switch is the cheapest dispatch.
function applyPairHeadBodyTail(a, b, p) {
  if (a.bond_inner === b || a.bond_outer === b) return 0;

  const rx = b.pos[0] - a.pos[0];
  const ry = b.pos[1] - a.pos[1];
  const rz = b.pos[2] - a.pos[2];
  const d2 = rx*rx + ry*ry + rz*rz;
  if (d2 > p.cutoff * p.cutoff) return 0;
  if (d2 < 1e-8) {
    const k = 5;
    const px = Math.random()-0.5, py = Math.random()-0.5, pz = Math.random()-0.5;
    a.force[0] -= px*k; a.force[1] -= py*k; a.force[2] -= pz*k;
    b.force[0] += px*k; b.force[1] += py*k; b.force[2] += pz*k;
    return 1;
  }
  const d = Math.sqrt(d2);
  const inv_d = 1 / d;
  const m  = smoothMask(d, p.cutoff);
  if (m === 0) return 1;
  const dm = smoothMaskDeriv(d, p.cutoff);

  const tlo = a.type < b.type ? a.type : b.type;
  const thi = a.type < b.type ? b.type : a.type;
  let want_a, want_rep, want_att;
  switch (tlo * 3 + thi) {
    case 0: want_a = p.hbt_hh_distance; want_rep = p.hbt_hh_repulsion; want_att = p.hbt_hh_attraction; break;
    case 1: want_a = p.hbt_hb_distance; want_rep = p.hbt_hb_repulsion; want_att = p.hbt_hb_attraction; break;
    case 2: {
      // Head↔tail interaction is restricted to the *same* chain — heads
      // and tails in different molecules don't feel each other (avoids
      // cross-shell coupling that wouldn't make physical sense for chains
      // tethered through their own body). The chain link: head's body
      // (head.bond_inner) is the same Block as tail's body (tail.bond_outer).
      const head = a.type === 0 ? a : b;
      const tail = a.type === 2 ? a : b;
      if (head.bond_inner === null || head.bond_inner !== tail.bond_outer) return 0;
      want_a = p.hbt_ht_distance; want_rep = p.hbt_ht_repulsion; want_att = p.hbt_ht_attraction;
      break;
    }
    case 4: want_a = p.hbt_bb_distance; want_rep = p.hbt_bb_repulsion; want_att = p.hbt_bb_attraction; break;
    case 5: want_a = p.hbt_bt_distance; want_rep = p.hbt_bt_repulsion; want_att = p.hbt_bt_attraction; break;
    default: want_a = p.hbt_tt_distance; want_rep = p.hbt_tt_repulsion; want_att = p.hbt_tt_attraction; break;
  }

  const alpha = a.normal[0]*b.normal[0] + a.normal[1]*b.normal[1] + a.normal[2]*b.normal[2];
  const cf = (1 + alpha) * 0.5;
  const da = d - want_a;
  const W  = 1 / (1 + da * da);
  const U0     = -want_rep * inv_d + cf * want_att * W;
  const dU0_dd =  want_rep * inv_d * inv_d - 2 * da * cf * want_att * W * W;

  const dU_dd = m * dU0_dd + dm * U0;
  const fmag = -dU_dd;
  const fx = fmag * rx * inv_d;
  const fy = fmag * ry * inv_d;
  const fz = fmag * rz * inv_d;
  a.force[0] -= fx; a.force[1] -= fy; a.force[2] -= fz;
  b.force[0] += fx; b.force[1] += fy; b.force[2] += fz;
  return 1;
}

// Wall: U_wall = wallK / (R - |pos|). Repels inward, blows up at the wall.
export function applyWall(b, p) {
  const r = Math.hypot(b.pos[0], b.pos[1], b.pos[2]);
  if (r < 1e-6) return;
  const margin = p.sphereR - r;
  let mag;
  if (margin <= 0.05) mag = -p.forceCap;
  else {
    mag = -p.wallK / (margin * margin);
    if (mag < -p.forceCap) mag = -p.forceCap;
  }
  const k = mag / r;
  b.force[0] += k * b.pos[0];
  b.force[1] += k * b.pos[1];
  b.force[2] += k * b.pos[2];
}

// Spatial hash over a uniform grid with cell size = cutoff. Two blocks can
// only interact if their cells differ by ≤ 1 in each axis, so for each cell
// we pair (a) every distinct in-cell pair and (b) every block with every
// block in 13 "forward" neighbor cells (the lexicographically-positive half
// of the 26 surrounding cells; the other half is covered when those cells
// are processed as the home cell).
const _FORWARD_OFFSETS = [
  [-1,-1, 1], [0,-1, 1], [1,-1, 1],
  [-1, 0, 1], [0, 0, 1], [1, 0, 1],
  [-1, 1, 1], [0, 1, 1], [1, 1, 1],
  [-1, 1, 0], [0, 1, 0], [1, 1, 0],
  [ 1, 0, 0],
];
// Pack (cx,cy,cz) ∈ [-512, 511] into a 30-bit integer for fast Map keys.
function _packCell(cx, cy, cz) {
  return ((cx + 512) << 20) | ((cy + 512) << 10) | (cz + 512);
}

// Semi-implicit Euler step over an array of blocks.
export function stepBlocks(blocks, dt, p) {
  for (const b of blocks) {
    b.force[0] = b.force[1] = b.force[2] = 0;
    b.torque[0] = b.torque[1] = b.torque[2] = 0;
    refreshNormal(b);
  }

  // Bond force/torque: applied unconditionally (no cutoff). Each bond is
  // stored as outer.bond_inner = inner, so iterating bond_inner over every
  // block hits each bond exactly once for either head_tail (1 bond/chain)
  // or head_body_tail (2 bonds/chain).
  if (p.config === 'head_tail' || p.config === 'head_body_tail') {
    for (let i = 0; i < blocks.length; i++) {
      const b = blocks[i];
      if (b.bond_inner !== null) applyBondForce(b, b.bond_inner, p);
    }
  }

  // Bin blocks into cells of size = cutoff. Track both the cell index per
  // block and the per-cell index list so we can iterate cells in any order.
  const cellSize = p.cutoff;
  const inv_cell = 1 / cellSize;
  const grid = new Map();              // packed cell key  →  number[] of block indices
  const cellsCx = [];                  // parallel arrays for cell decode without string parsing
  const cellsCy = [];
  const cellsCz = [];
  const cellsKey = [];
  const N = blocks.length;
  for (let i = 0; i < N; i++) {
    const pos = blocks[i].pos;
    const cx = Math.floor(pos[0] * inv_cell);
    const cy = Math.floor(pos[1] * inv_cell);
    const cz = Math.floor(pos[2] * inv_cell);
    const key = _packCell(cx, cy, cz);
    let cell = grid.get(key);
    if (cell === undefined) {
      cell = [];
      grid.set(key, cell);
      cellsCx.push(cx); cellsCy.push(cy); cellsCz.push(cz); cellsKey.push(key);
    }
    cell.push(i);
  }

  // Counts: pairs the spatial hash put through applyPair, and the subset that
  // actually got past the d² ≤ cutoff² test (i.e. contributed force/torque).
  let pairsChecked = 0;
  let pairsComputed = 0;

  const M = cellsKey.length;
  for (let c = 0; c < M; c++) {
    const cell = grid.get(cellsKey[c]);
    const cx = cellsCx[c], cy = cellsCy[c], cz = cellsCz[c];
    // (a) in-cell pairs
    for (let ii = 0; ii < cell.length; ii++) {
      const a = blocks[cell[ii]];
      for (let jj = ii + 1; jj < cell.length; jj++) {
        pairsChecked++;
        pairsComputed += applyPair(a, blocks[cell[jj]], p);
      }
    }
    // (b) cross-cell pairs to forward neighbors
    for (let o = 0; o < _FORWARD_OFFSETS.length; o++) {
      const off = _FORWARD_OFFSETS[o];
      const ncell = grid.get(_packCell(cx + off[0], cy + off[1], cz + off[2]));
      if (ncell === undefined) continue;
      for (let ii = 0; ii < cell.length; ii++) {
        const a = blocks[cell[ii]];
        for (let jj = 0; jj < ncell.length; jj++) {
          pairsChecked++;
          pairsComputed += applyPair(a, blocks[ncell[jj]], p);
        }
      }
    }
  }

  for (const b of blocks) applyWall(b, p);

  const fc = p.forceCap, fc2 = fc * fc;
  for (const b of blocks) {
    let m2 = b.force[0]**2 + b.force[1]**2 + b.force[2]**2;
    if (m2 > fc2) {
      const s = fc / Math.sqrt(m2);
      b.force[0] *= s; b.force[1] *= s; b.force[2] *= s;
    }
    m2 = b.torque[0]**2 + b.torque[1]**2 + b.torque[2]**2;
    if (m2 > fc2) {
      const s = fc / Math.sqrt(m2);
      b.torque[0] *= s; b.torque[1] *= s; b.torque[2] *= s;
    }

    const damp = p.damping;
    for (let k = 0; k < 3; k++) {
      b.vel[k] = (b.vel[k] + b.force[k] * dt) * damp;
      b.pos[k] += b.vel[k] * dt;
      b.ang[k] = (b.ang[k] + b.torque[k] * dt) * damp;
    }

    // Quaternion exponential update for world-frame angular velocity.
    const wx = b.ang[0]*dt*0.5, wy = b.ang[1]*dt*0.5, wz = b.ang[2]*dt*0.5;
    const half = Math.hypot(wx, wy, wz);
    if (half > 1e-8) {
      const s = Math.sin(half) / half;
      const dx = wx*s, dy = wy*s, dz = wz*s, dw = Math.cos(half);
      const qx = b.quat[0], qy = b.quat[1], qz = b.quat[2], qw = b.quat[3];
      // q_new = dq * q
      let nx = dw*qx + dx*qw + dy*qz - dz*qy;
      let ny = dw*qy - dx*qz + dy*qw + dz*qx;
      let nz = dw*qz + dx*qy - dy*qx + dz*qw;
      let nw = dw*qw - dx*qx - dy*qy - dz*qz;
      const inv = 1 / Math.hypot(nx, ny, nz, nw);
      b.quat[0] = nx*inv; b.quat[1] = ny*inv; b.quat[2] = nz*inv; b.quat[3] = nw*inv;
    }
  }

  return { pairsChecked, pairsComputed };
}

// ---------------------------------------------------------------------------
// Random initial state helpers.
// ---------------------------------------------------------------------------
export function randomQuat() {
  const u1 = Math.random(), u2 = Math.random(), u3 = Math.random();
  const s1 = Math.sqrt(1 - u1), s2 = Math.sqrt(u1);
  return [
    s1 * Math.sin(2 * Math.PI * u2),
    s1 * Math.cos(2 * Math.PI * u2),
    s2 * Math.sin(2 * Math.PI * u3),
    s2 * Math.cos(2 * Math.PI * u3),
  ];
}
export function randomPointInBall(R) {
  for (;;) {
    const x = (Math.random()*2 - 1) * R;
    const y = (Math.random()*2 - 1) * R;
    const z = (Math.random()*2 - 1) * R;
    if (x*x + y*y + z*z <= R*R) return [x, y, z];
  }
}
