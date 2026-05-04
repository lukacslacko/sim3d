// Orientation-alignment grid: an N×N×N lattice of coordinate frames inside
// a periodic cube. Positions are fixed; only orientations evolve. Each step,
// every frame rotates a small fraction `eta` of the way toward the average
// orientation of its 6 face-adjacent neighbors (with wrap). Optional Gaussian
// noise per step prevents the system from locking on metastable defects.
//
// State layout (Float32Array typed, indexed (iz·N + iy)·N + ix):
//   positions: 3 floats per cell — fixed grid points centered at (-L/2, L/2)³.
//   quats:     4 floats per cell — orientation [x, y, z, w].
//   scratch:   4 floats per cell — write target each step (then swap).
//
// All math is local: no THREE.js dependency, so this can run under node for
// tests and is cheap to call from the render loop.

const NEIGHBOR_OFFSETS = [
  [-1, 0, 0], [1, 0, 0],
  [0, -1, 0], [0, 1, 0],
  [0, 0, -1], [0, 0, 1],
];

export function createAlignGrid(N, L) {
  const total = N * N * N;
  const state = {
    N, L, s: L / N,
    positions: new Float32Array(total * 3),
    quats:     new Float32Array(total * 4),
    scratch:   new Float32Array(total * 4),
  };
  const half = 0.5 * L;
  const s = state.s;
  for (let iz = 0; iz < N; iz++) {
    for (let iy = 0; iy < N; iy++) {
      for (let ix = 0; ix < N; ix++) {
        const i = (iz * N + iy) * N + ix;
        // Cell-centered grid points so the outer cells sit a half-spacing
        // off each face; under PBC this matches the spacing between
        // wrap-adjacent cells.
        state.positions[3*i    ] = (ix + 0.5) * s - half;
        state.positions[3*i + 1] = (iy + 0.5) * s - half;
        state.positions[3*i + 2] = (iz + 0.5) * s - half;
      }
    }
  }
  randomizeAlignGrid(state);
  return state;
}

// Replace every cell's quaternion with a uniform random rotation (Shoemake).
export function randomizeAlignGrid(state) {
  const { N, quats } = state;
  const total = N * N * N;
  for (let i = 0; i < total; i++) {
    const u1 = Math.random(), u2 = Math.random(), u3 = Math.random();
    const r1 = Math.sqrt(1 - u1), r2 = Math.sqrt(u1);
    quats[4*i    ] = r1 * Math.sin(2 * Math.PI * u2);
    quats[4*i + 1] = r1 * Math.cos(2 * Math.PI * u2);
    quats[4*i + 2] = r2 * Math.sin(2 * Math.PI * u3);
    quats[4*i + 3] = r2 * Math.cos(2 * Math.PI * u3);
  }
}

// One alignment step. For each cell:
//   1) Sum the 6 face-adjacent neighbors' quaternions, sign-fixing each so
//      it lies in the same hemisphere as the self quaternion (q and -q
//      represent the same rotation; without the sign-fix the average of two
//      antipodal-but-identical rotations would cancel to zero).
//   2) Normalize the sum: that's the (extrinsic) average orientation.
//   3) Linearly interpolate self → average by η, renormalize. Lerp+normalize
//      is an excellent approximation to slerp for small η and avoids trig.
//   4) Optional per-axis quaternion noise to anneal out defects.
// Writes into a scratch buffer, then swaps — reads-from-old, writes-to-new
// keeps the update symmetric across all cells in the same step.
//
// Noise: each component of the new quaternion gets a uniform random kick in
// [-noise, noise] before renormalization. Acts like temperature: high noise
// keeps the system fluctuating; setting noise → 0 lets it freeze.
export function stepAlign(state, eta, noise) {
  const { N, quats, scratch } = state;
  for (let iz = 0; iz < N; iz++) {
    for (let iy = 0; iy < N; iy++) {
      for (let ix = 0; ix < N; ix++) {
        const i = (iz * N + iy) * N + ix;
        const qx = quats[4*i    ];
        const qy = quats[4*i + 1];
        const qz = quats[4*i + 2];
        const qw = quats[4*i + 3];

        let sx = 0, sy = 0, sz = 0, sw = 0;
        for (let nb = 0; nb < 6; nb++) {
          const off = NEIGHBOR_OFFSETS[nb];
          const jx = ((ix + off[0]) % N + N) % N;
          const jy = ((iy + off[1]) % N + N) % N;
          const jz = ((iz + off[2]) % N + N) % N;
          const j = (jz * N + jy) * N + jx;
          let nqx = quats[4*j    ];
          let nqy = quats[4*j + 1];
          let nqz = quats[4*j + 2];
          let nqw = quats[4*j + 3];
          // q and -q are the same rotation in SO(3); pick the copy that's
          // on the same side of S³ as self before adding to the sum.
          if (nqx*qx + nqy*qy + nqz*qz + nqw*qw < 0) {
            nqx = -nqx; nqy = -nqy; nqz = -nqz; nqw = -nqw;
          }
          sx += nqx; sy += nqy; sz += nqz; sw += nqw;
        }
        const len = Math.hypot(sx, sy, sz, sw);
        if (len > 1e-8) {
          const inv = 1 / len;
          sx *= inv; sy *= inv; sz *= inv; sw *= inv;
        } else {
          // Average cancelled to zero (extremely unlikely with sign-fix);
          // fall back to no movement so the step is well-defined.
          sx = qx; sy = qy; sz = qz; sw = qw;
        }
        // Lerp toward avg, then renormalize — a slerp approximation.
        let nx = (1 - eta) * qx + eta * sx;
        let ny = (1 - eta) * qy + eta * sy;
        let nz = (1 - eta) * qz + eta * sz;
        let nw = (1 - eta) * qw + eta * sw;
        if (noise > 0) {
          nx += (Math.random() * 2 - 1) * noise;
          ny += (Math.random() * 2 - 1) * noise;
          nz += (Math.random() * 2 - 1) * noise;
          nw += (Math.random() * 2 - 1) * noise;
        }
        const inv = 1 / Math.hypot(nx, ny, nz, nw);
        scratch[4*i    ] = nx * inv;
        scratch[4*i + 1] = ny * inv;
        scratch[4*i + 2] = nz * inv;
        scratch[4*i + 3] = nw * inv;
      }
    }
  }
  // Swap buffers — next step reads what we just wrote.
  state.quats = scratch;
  state.scratch = quats;
}

// Fill out[] (length N³ · 18) with vertex pairs for line segments rendering
// each frame's three coordinate axes from its grid point outward. Vertex
// layout per cell: [origin_X, end_X, origin_Y, end_Y, origin_Z, end_Z].
// Pair this with a one-time per-vertex color buffer painting X red, Y green,
// Z blue — that's the "render the rgb xyz coordinates" visualization.
//
// offsetCells (optional [ox, oy, oz] integers): shifts every cell's display
// position by that many grid spacings along each cube axis, wrapping back
// into [-L/2, L/2). Since the topology is periodic (opposite faces are
// glued), this is a pure render shift — the simulation state is unchanged.
export function fillAxesGeometry(state, axisLen, out, offsetCells) {
  const { N, L, s, positions, quats } = state;
  const half = 0.5 * L;
  const offX = offsetCells ? offsetCells[0] * s : 0;
  const offY = offsetCells ? offsetCells[1] * s : 0;
  const offZ = offsetCells ? offsetCells[2] * s : 0;
  const total = N * N * N;
  for (let i = 0; i < total; i++) {
    let px = positions[3*i    ] + offX;
    let py = positions[3*i + 1] + offY;
    let pz = positions[3*i + 2] + offZ;
    // Wrap into [-L/2, L/2) using floor-div, same convention as the cube
    // PBC in physics.mjs. No-op when offset is (0,0,0).
    px -= L * Math.floor((px + half) / L);
    py -= L * Math.floor((py + half) / L);
    pz -= L * Math.floor((pz + half) / L);
    const qx = quats[4*i    ];
    const qy = quats[4*i + 1];
    const qz = quats[4*i + 2];
    const qw = quats[4*i + 3];
    // Rotation matrix columns = R·(1,0,0), R·(0,1,0), R·(0,0,1).
    const xx = 1 - 2*(qy*qy + qz*qz);
    const xy = 2*(qx*qy + qz*qw);
    const xz = 2*(qx*qz - qy*qw);
    const yx = 2*(qx*qy - qz*qw);
    const yy = 1 - 2*(qx*qx + qz*qz);
    const yz = 2*(qy*qz + qx*qw);
    const zx = 2*(qx*qz + qy*qw);
    const zy = 2*(qy*qz - qx*qw);
    const zz = 1 - 2*(qx*qx + qy*qy);
    const ci = i * 18;
    out[ci     ] = px;              out[ci +  1] = py;              out[ci +  2] = pz;
    out[ci +  3] = px + xx*axisLen; out[ci +  4] = py + xy*axisLen; out[ci +  5] = pz + xz*axisLen;
    out[ci +  6] = px;              out[ci +  7] = py;              out[ci +  8] = pz;
    out[ci +  9] = px + yx*axisLen; out[ci + 10] = py + yy*axisLen; out[ci + 11] = pz + yz*axisLen;
    out[ci + 12] = px;              out[ci + 13] = py;              out[ci + 14] = pz;
    out[ci + 15] = px + zx*axisLen; out[ci + 16] = py + zy*axisLen; out[ci + 17] = pz + zz*axisLen;
  }
}

// Fill out[] (length N³ · 3) with each frame's +Z axis world direction.
// Used as the input to the spherical histogram (one direction per cell).
export function fillZDirections(state, out) {
  const { N, quats } = state;
  const total = N * N * N;
  for (let i = 0; i < total; i++) {
    const qx = quats[4*i    ];
    const qy = quats[4*i + 1];
    const qz = quats[4*i + 2];
    const qw = quats[4*i + 3];
    out[3*i    ] = 2*(qx*qz + qy*qw);
    out[3*i + 1] = 2*(qy*qz - qx*qw);
    out[3*i + 2] = 1 - 2*(qx*qx + qy*qy);
  }
}

// Per-cell mean misalignment to its 6 face-adjacent neighbors. Writes to
// out[] (length N³). Per neighbor we use 1 - (q · q')² ∈ [0, 1]:
//   • 0  → q and q' represent the same SO(3) rotation (q' = ±q).
//   • 1  → q and q' are 4D-orthogonal, i.e. 180° apart in SO(3).
// Averaged over the 6 face-adjacent neighbors (with PBC). For a uniform-
// random grid each cell has mean ~0.75; a perfectly aligned grid gives 0.
// The visual "voxel" rendering colors each grid point by this scalar so
// you can see where the system has converged and where defects persist.
export function fillCellMisalignment(state, out) {
  const { N, quats } = state;
  for (let iz = 0; iz < N; iz++) {
    for (let iy = 0; iy < N; iy++) {
      for (let ix = 0; ix < N; ix++) {
        const i = (iz * N + iy) * N + ix;
        const qx = quats[4*i    ];
        const qy = quats[4*i + 1];
        const qz = quats[4*i + 2];
        const qw = quats[4*i + 3];
        let sum = 0;
        for (let nb = 0; nb < 6; nb++) {
          const off = NEIGHBOR_OFFSETS[nb];
          const jx = ((ix + off[0]) % N + N) % N;
          const jy = ((iy + off[1]) % N + N) % N;
          const jz = ((iz + off[2]) % N + N) % N;
          const j = (jz * N + jy) * N + jx;
          const dot = qx*quats[4*j    ]
                    + qy*quats[4*j + 1]
                    + qz*quats[4*j + 2]
                    + qw*quats[4*j + 3];
          sum += 1 - dot*dot;
        }
        out[i] = sum * (1 / 6);
      }
    }
  }
}

// Optional diagnostic: average of (1 - cos(θ_ij/2)²) across every face-
// adjacent pair, where θ_ij is the rotation angle from cell i to cell j.
// Returns 0 for a perfectly aligned grid, ~0.5 for uniformly random.
export function meanNeighborMisalignment(state) {
  const { N, quats } = state;
  let sum = 0, count = 0;
  for (let iz = 0; iz < N; iz++) {
    for (let iy = 0; iy < N; iy++) {
      for (let ix = 0; ix < N; ix++) {
        const i = (iz * N + iy) * N + ix;
        const qx = quats[4*i], qy = quats[4*i+1], qz = quats[4*i+2], qw = quats[4*i+3];
        // Only the +x, +y, +z neighbors so each pair is counted once.
        for (let d = 0; d < 3; d++) {
          const jx = d === 0 ? (ix + 1) % N : ix;
          const jy = d === 1 ? (iy + 1) % N : iy;
          const jz = d === 2 ? (iz + 1) % N : iz;
          const j = (jz * N + jy) * N + jx;
          const nqx = quats[4*j], nqy = quats[4*j+1], nqz = quats[4*j+2], nqw = quats[4*j+3];
          // |q · q'| = |cos(θ/2)|; misalignment = 1 - that².
          const dot = qx*nqx + qy*nqy + qz*nqz + qw*nqw;
          sum += 1 - dot*dot;
          count++;
        }
      }
    }
  }
  return count > 0 ? sum / count : 0;
}
