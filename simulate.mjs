#!/usr/bin/env node
// Headless run of the sim3d physics. Prints distance / nearest-neighbor
// statistics so we can verify the affinity model behaves the way we expect.
//
// Usage:
//   node simulate.mjs                    # default: new (LJ-style) model
//   node simulate.mjs --model=old        # original 1/(1+(d-1)^2) bump
//   node simulate.mjs --steps=4000 --n=80 --R=8 --seed=1

import {
  DEFAULTS, makeBlock, refreshNormal, applyPair, applyWall,
  pairAffinity, pairAffinityGrad, stepBlocks,
  randomQuat, randomPointInBall,
} from './physics.mjs';

// ---- arg parsing -----------------------------------------------------------
const args = Object.fromEntries(process.argv.slice(2)
  .map(a => a.replace(/^--/, '').split('='))
  .map(([k, v]) => [k, v ?? 'true']));
const N      = +(args.n ?? 80);
const STEPS  = +(args.steps ?? 1500);
const DT     = +(args.dt ?? 0.04);
const SEED   = +(args.seed ?? 1);
const MODEL  = args.model ?? 'new';

// Deterministic PRNG so runs are reproducible.
let _seed = SEED >>> 0;
const _rand = () => { _seed = (_seed * 1664525 + 1013904223) >>> 0; return _seed / 0x100000000; };
Math.random = _rand;

const params = { ...DEFAULTS, sphereR: +(args.R ?? 8) };

// ---- optional: monkey-patch the affinity to the OLD model for comparison ---
if (MODEL === 'old') {
  // U_old(d, cA, cB) = -1/d + s2 * 1/(1+(d-1)^2)
  // ∂U/∂d  = 1/d² - s2 * 2(d-1)/(1+(d-1)²)²
  // ∂U/∂cA = -cA / (1+(d-1)²)
  globalThis.__OLD = true;
  // Replace functions in the imported module's namespace by wrapping applyPair
  // ourselves; the simplest way is to re-implement the integrator here.
}

// ---- init ------------------------------------------------------------------
const blocks = [];
for (let i = 0; i < N; i++) {
  blocks.push(makeBlock(randomPointInBall(params.sphereR * 0.9), randomQuat()));
}

// ---- old-model variant: a local integrator that uses the old gradients ----
function applyPairOld(a, b, p) {
  const rx = b.pos[0]-a.pos[0], ry = b.pos[1]-a.pos[1], rz = b.pos[2]-a.pos[2];
  const d = Math.hypot(rx, ry, rz);
  if (d < 1e-4) return;
  const inv_d = 1/d;
  const hx = rx*inv_d, hy = ry*inv_d, hz = rz*inv_d;
  const nA = a.normal, nB = b.normal;
  const cA = hx*nA[0] + hy*nA[1] + hz*nA[2];
  const cB = hx*nB[0] + hy*nB[1] + hz*nB[2];
  const dm1 = d - 1, denom = 1 + dm1*dm1;
  const s2 = 1 - 0.5*(cA*cA + cB*cB);
  const dU_dd  = 1/(d*d) - s2 * 2*dm1/(denom*denom);
  const dU_dcA = -cA / denom;
  const dU_dcB = -cB / denom;
  const ka = -dU_dcA*inv_d, kb = -dU_dcB*inv_d;
  const fbx = -dU_dd*hx + ka*(nA[0]-cA*hx) + kb*(nB[0]-cB*hx);
  const fby = -dU_dd*hy + ka*(nA[1]-cA*hy) + kb*(nB[1]-cB*hy);
  const fbz = -dU_dd*hz + ka*(nA[2]-cA*hz) + kb*(nB[2]-cB*hz);
  a.force[0] -= fbx; a.force[1] -= fby; a.force[2] -= fbz;
  b.force[0] += fbx; b.force[1] += fby; b.force[2] += fbz;
  // torques
  const tax = nA[1]*hz - nA[2]*hy, tay = nA[2]*hx - nA[0]*hz, taz = nA[0]*hy - nA[1]*hx;
  a.torque[0] -= dU_dcA*tax; a.torque[1] -= dU_dcA*tay; a.torque[2] -= dU_dcA*taz;
  const tbx = nB[1]*hz - nB[2]*hy, tby = nB[2]*hx - nB[0]*hz, tbz = nB[0]*hy - nB[1]*hx;
  b.torque[0] -= dU_dcB*tbx; b.torque[1] -= dU_dcB*tby; b.torque[2] -= dU_dcB*tbz;
}

function stepOld(blocks, dt, p) {
  for (const b of blocks) {
    b.force[0]=b.force[1]=b.force[2]=0;
    b.torque[0]=b.torque[1]=b.torque[2]=0;
    refreshNormal(b);
  }
  for (let i=0;i<blocks.length;i++) for (let j=i+1;j<blocks.length;j++) applyPairOld(blocks[i], blocks[j], p);
  for (const b of blocks) applyWall(b, p);
  // reuse the integrator step from physics.mjs by inlining the rest:
  const fc=p.forceCap, fc2=fc*fc;
  for (const b of blocks) {
    let m2 = b.force[0]**2 + b.force[1]**2 + b.force[2]**2;
    if (m2 > fc2) { const s=fc/Math.sqrt(m2); b.force[0]*=s; b.force[1]*=s; b.force[2]*=s; }
    m2 = b.torque[0]**2 + b.torque[1]**2 + b.torque[2]**2;
    if (m2 > fc2) { const s=fc/Math.sqrt(m2); b.torque[0]*=s; b.torque[1]*=s; b.torque[2]*=s; }
    const damp = p.damping;
    for (let k=0;k<3;k++) {
      b.vel[k] = (b.vel[k] + b.force[k]*dt) * damp;
      b.pos[k] += b.vel[k]*dt;
      b.ang[k] = (b.ang[k] + b.torque[k]*dt) * damp;
    }
    const wx=b.ang[0]*dt*0.5, wy=b.ang[1]*dt*0.5, wz=b.ang[2]*dt*0.5;
    const half = Math.hypot(wx,wy,wz);
    if (half > 1e-8) {
      const s=Math.sin(half)/half;
      const dx=wx*s, dy=wy*s, dz=wz*s, dw=Math.cos(half);
      const qx=b.quat[0], qy=b.quat[1], qz=b.quat[2], qw=b.quat[3];
      let nx=dw*qx + dx*qw + dy*qz - dz*qy;
      let ny=dw*qy - dx*qz + dy*qw + dz*qx;
      let nz=dw*qz + dx*qy - dy*qx + dz*qw;
      let nw=dw*qw - dx*qx - dy*qy - dz*qz;
      const inv=1/Math.hypot(nx,ny,nz,nw);
      b.quat[0]=nx*inv; b.quat[1]=ny*inv; b.quat[2]=nz*inv; b.quat[3]=nw*inv;
    }
  }
}

// ---- statistics ------------------------------------------------------------
function pct(arr, p) {
  const i = Math.min(arr.length - 1, Math.max(0, Math.floor(arr.length * p)));
  return arr[i];
}
function stats(blocks) {
  const dists = [];
  const nn = new Array(blocks.length).fill(Infinity);
  for (let i = 0; i < blocks.length; i++) {
    for (let j = i + 1; j < blocks.length; j++) {
      const dx = blocks[i].pos[0]-blocks[j].pos[0];
      const dy = blocks[i].pos[1]-blocks[j].pos[1];
      const dz = blocks[i].pos[2]-blocks[j].pos[2];
      const d = Math.sqrt(dx*dx + dy*dy + dz*dz);
      dists.push(d);
      if (d < nn[i]) nn[i] = d;
      if (d < nn[j]) nn[j] = d;
    }
  }
  dists.sort((a, b) => a - b);
  nn.sort((a, b) => a - b);
  const radii = blocks.map(b => Math.hypot(b.pos[0], b.pos[1], b.pos[2])).sort((a, b) => a - b);

  // also: cosine alignment of nearest neighbors (proxy for how parallel sheets are)
  const aligns = [];
  for (let i = 0; i < blocks.length; i++) {
    let best = -1, bestD = Infinity;
    for (let j = 0; j < blocks.length; j++) {
      if (i === j) continue;
      const dx = blocks[i].pos[0]-blocks[j].pos[0];
      const dy = blocks[i].pos[1]-blocks[j].pos[1];
      const dz = blocks[i].pos[2]-blocks[j].pos[2];
      const d = Math.sqrt(dx*dx + dy*dy + dz*dz);
      if (d < bestD) { bestD = d; best = j; }
    }
    if (best >= 0) {
      const ai = blocks[i].normal, bj = blocks[best].normal;
      aligns.push(Math.abs(ai[0]*bj[0] + ai[1]*bj[1] + ai[2]*bj[2]));
    }
  }
  aligns.sort((a,b) => a-b);

  return {
    pairD_p01: dists[0]?.toFixed(3),
    pairD_p10: pct(dists, 0.10)?.toFixed(3),
    pairD_p50: pct(dists, 0.50)?.toFixed(3),
    nn_min:    nn[0]?.toFixed(3),
    nn_p50:    pct(nn, 0.5)?.toFixed(3),
    nn_max:    nn[nn.length-1]?.toFixed(3),
    radius_p50:radii[Math.floor(radii.length/2)]?.toFixed(3),
    radius_max:radii[radii.length-1]?.toFixed(3),
    alignNN_p50: pct(aligns, 0.5)?.toFixed(3),
  };
}

// ---- run -------------------------------------------------------------------
const stepFn = (MODEL === 'old') ? stepOld : stepBlocks;
console.log(`model=${MODEL}  N=${N}  R=${params.sphereR}  steps=${STEPS}  dt=${DT}  seed=${SEED}`);
console.log('keys: pairD_p01 = 1st-percentile pair distance (smallest gap), nn = nearest-neighbor distance, alignNN = |n_i · n_nn| (1=parallel)');
console.log('start:', stats(blocks));

const t0 = performance.now();
for (let s = 1; s <= STEPS; s++) {
  stepFn(blocks, DT, params);
  if (s % Math.max(1, STEPS / 6 | 0) === 0) {
    console.log(`step ${String(s).padStart(5)}:`, stats(blocks));
  }
}
const t1 = performance.now();
console.log(`done in ${(t1-t0).toFixed(0)} ms`);
