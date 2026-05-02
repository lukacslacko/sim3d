#!/usr/bin/env node
// Sanity check: compare the analytical gradient in pairAffinityGrad to a
// central finite-difference approximation of pairAffinity, across a sweep of
// (d, cA, cB) values. Prints max relative error.

import { pairAffinity, pairAffinityGrad, DEFAULTS } from './physics.mjs';

const p = { ...DEFAULTS };
const eps = 1e-5;

let worst = 0, worstAt = '';
function check(d, cA, cB) {
  const [adU_dd, adU_dcA, adU_dcB] = pairAffinityGrad(d, cA, cB, p);
  const fdU_dd  = (pairAffinity(d+eps, cA, cB, p) - pairAffinity(d-eps, cA, cB, p)) / (2*eps);
  const fdU_dcA = (pairAffinity(d, cA+eps, cB, p) - pairAffinity(d, cA-eps, cB, p)) / (2*eps);
  const fdU_dcB = (pairAffinity(d, cA, cB+eps, p) - pairAffinity(d, cA, cB-eps, p)) / (2*eps);

  for (const [ana, fd, name] of [
    [adU_dd, fdU_dd, '∂U/∂d'],
    [adU_dcA, fdU_dcA, '∂U/∂cA'],
    [adU_dcB, fdU_dcB, '∂U/∂cB'],
  ]) {
    // mixed tolerance: rel for big values, abs floor for near-zero ones
    const denom = Math.abs(ana) + Math.abs(fd) + 1e-3;
    const rel = Math.abs(ana - fd) / denom;
    if (rel > worst) { worst = rel; worstAt = `${name}@(d=${d}, cA=${cA}, cB=${cB})  ana=${ana.toExponential(3)}  fd=${fd.toExponential(3)}`; }
  }
}

const ds  = [0.3, 0.7, 1.0, 1.4, 2.0, 4.0, 8.0];
const cs  = [-0.95, -0.5, -0.1, 0.0, 0.1, 0.5, 0.95];
for (const d of ds) for (const cA of cs) for (const cB of cs) check(d, cA, cB);

console.log(`max relative gradient error over ${ds.length*cs.length*cs.length} configs: ${worst.toExponential(3)}`);
console.log(`worst at: ${worstAt}`);
console.log(worst < 1e-5 ? 'OK — analytical gradient matches finite differences.' : 'FAIL — gradient mismatch.');
