// Shared physics for the sim3d playground.
// Pure JS — no THREE dependency, so this also runs under node for analysis.
//
// All vectors are length-3 number[] (mutable). Quaternions are [x,y,z,w].
// A block has its sheet normal along its local +Z axis.

export const DEFAULTS = {
  sphereR: 10,
  damping: 0.92,
  wallK: 1.0,
  forceCap: 50,
  // Pair affinity parameters — see pairAffinity().
  // Sign convention: the pair energy is built around -1/d, so for a parameter
  // to act as an actual repulsion (energy → +∞ as d → 0) it should be NEGATIVE
  // (since -1/d × negative = +positive/d). Likewise `attraction` should be
  // negative to make the well at d_peak an actual energy minimum.
  in_plane_repulsion:   -0.7,
  orthogonal_repulsion: -1.4,
  attraction:           -0.4,
  d_peak:                0.75,
};

export function makeBlock(pos, quat) {
  return {
    pos:    [pos[0], pos[1], pos[2]],
    quat:   [quat[0], quat[1], quat[2], quat[3]],
    vel:    [0, 0, 0],
    ang:    [0, 0, 0],
    force:  [0, 0, 0],
    torque: [0, 0, 0],
    normal: [0, 0, 0],
  };
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

export function pairAffinity(d, cA, cB, p) {
  const T = 0.5 * (cA*cA + cB*cB);
  const S = 1 - T;
  const dpk = d - p.d_peak;
  const W = 1 / (1 + dpk * dpk);
  const pre = p.in_plane_repulsion + (p.orthogonal_repulsion - p.in_plane_repulsion) * T;
  return -pre / d + S * p.attraction * W;
}

// Returns [∂U/∂d, ∂U/∂cA, ∂U/∂cB].
//
//   ∂U/∂d  =  pre/d²  +  S · A · dW/dd            with dW/dd = −2(d−d_peak)·W²
//   ∂U/∂T  =  −(P_ortho − P_in)/d  −  A · W       (since ∂S/∂T = −1)
//   ∂T/∂cA =  cA           so  ∂U/∂cA = cA · ∂U/∂T   (and similarly for cB)
export function pairAffinityGrad(d, cA, cB, p) {
  const T = 0.5 * (cA*cA + cB*cB);
  const S = 1 - T;
  const dpk = d - p.d_peak;
  const W = 1 / (1 + dpk * dpk);
  const dW_dd = -2 * dpk * W * W;
  const pre = p.in_plane_repulsion + (p.orthogonal_repulsion - p.in_plane_repulsion) * T;

  const dU_dd = pre / (d * d) + S * p.attraction * dW_dd;
  const dU_dT = -(p.orthogonal_repulsion - p.in_plane_repulsion) / d - p.attraction * W;
  const dU_dcA = cA * dU_dT;
  const dU_dcB = cB * dU_dT;
  return [dU_dd, dU_dcA, dU_dcB];
}

// Accumulate pair force on a, b and torques (about each block's own normal).
const _r = [0,0,0], _rhat = [0,0,0], _fb = [0,0,0];
export function applyPair(a, b, p) {
  _r[0] = b.pos[0] - a.pos[0];
  _r[1] = b.pos[1] - a.pos[1];
  _r[2] = b.pos[2] - a.pos[2];
  let d = Math.hypot(_r[0], _r[1], _r[2]);
  if (d < 1e-4) {
    // Degenerate overlap — kick apart along a random axis to break symmetry.
    const k = 5;
    const rx = Math.random()-0.5, ry = Math.random()-0.5, rz = Math.random()-0.5;
    a.force[0] -= rx*k; a.force[1] -= ry*k; a.force[2] -= rz*k;
    b.force[0] += rx*k; b.force[1] += ry*k; b.force[2] += rz*k;
    return;
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

// Semi-implicit Euler step over an array of blocks.
export function stepBlocks(blocks, dt, p) {
  for (const b of blocks) {
    b.force[0] = b.force[1] = b.force[2] = 0;
    b.torque[0] = b.torque[1] = b.torque[2] = 0;
    refreshNormal(b);
  }
  const N = blocks.length;
  for (let i = 0; i < N; i++)
    for (let j = i + 1; j < N; j++) applyPair(blocks[i], blocks[j], p);
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
