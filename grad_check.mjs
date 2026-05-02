#!/usr/bin/env node
// Sanity check: compare the analytical gradient in pairAffinityGrad to a
// central finite-difference approximation of pairAffinity, across a sweep of
// (d, cA, cB) values. Prints max relative error.

import { pairAffinity, pairAffinityGrad, DEFAULTS } from './physics.mjs';

// Run the gradient check at multiple bend values so the bent-angle path
// (and the √(1−c²) derivative) gets exercised, not just the bend=0 fast path.
const BENDS = [0, 0.05, 0.2, -0.15];
const BASE = { ...DEFAULTS };
const eps = 1e-5;

let worst = 0, worstAt = '';
function check(d, cA, cB, p) {
  const [adU_dd, adU_dcA, adU_dcB] = pairAffinityGrad(d, cA, cB, p);
  const fdU_dd  = (pairAffinity(d+eps, cA, cB, p) - pairAffinity(d-eps, cA, cB, p)) / (2*eps);
  const fdU_dcA = (pairAffinity(d, cA+eps, cB, p) - pairAffinity(d, cA-eps, cB, p)) / (2*eps);
  const fdU_dcB = (pairAffinity(d, cA, cB+eps, p) - pairAffinity(d, cA, cB-eps, p)) / (2*eps);

  for (const [ana, fd, name] of [
    [adU_dd, fdU_dd, '∂U/∂d'],
    [adU_dcA, fdU_dcA, '∂U/∂cA'],
    [adU_dcB, fdU_dcB, '∂U/∂cB'],
  ]) {
    // Tolerance: we accept either small relative error (for non-trivial values)
    // or small absolute error (handles points exactly at the smoothstep
    // boundary, where the FD picks up O(ε) noise from the C² discontinuity).
    const absErr = Math.abs(ana - fd);
    if (absErr < 1e-4) continue;
    const denom = Math.abs(ana) + Math.abs(fd) + 1e-3;
    const rel = absErr / denom;
    if (rel > worst) { worst = rel; worstAt = `${name}@(bend=${p.bend}, d=${d}, cA=${cA}, cB=${cB})  ana=${ana.toExponential(3)}  fd=${fd.toExponential(3)}`; }
  }
}

// Spread d across the three regimes of the smooth cutoff: full strength
// (d ≤ cutoff/2), taper region (cutoff/2 < d < cutoff), past cutoff (d ≥ cutoff).
const ds  = [0.3, 0.7, 1.0, 1.4, 1.55, 1.8, 2.0, 2.5, 2.9, 3.1, 4.0, 8.0];
const cs  = [-0.95, -0.5, -0.1, 0.0, 0.1, 0.5, 0.95];
let total = 0;
for (const bend of BENDS) {
  const p = { ...BASE, bend };
  for (const d of ds) for (const cA of cs) for (const cB of cs) { check(d, cA, cB, p); total++; }
}

console.log(`max relative gradient error over ${total} configs (${BENDS.length} bend values): ${worst.toExponential(3)}`);
console.log(`worst at: ${worstAt}`);
console.log(worst < 1e-5 ? 'OK — analytical gradient matches finite differences.' : 'FAIL — gradient mismatch.');
