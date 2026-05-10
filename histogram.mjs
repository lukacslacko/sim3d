// Spherical direction histogram, factored out of index.html so multiple
// pages can share it. Bins unit vectors onto an icosahedron's faces (each
// face = one bin), lifts denser bins radially outward, and bridges adjacent
// bins with vertical walls so the surface reads as a connected step plot
// instead of a field of floating triangles.
//
// Usage:
//   const hist = createDirectionHistogram(canvasEl);
//   hist.update(dirsFloat32, count);   // count unit vectors, 3 floats each
//   hist.syncCamera(mainCamera);       // align mini-camera with main view
//   hist.render();

import * as THREE from 'three';

const HIST_SUBDIV = 3;             // subdivision-3 icosahedron → 1280 faces
const HIST_RADIUS_BOOST = 0.6;     // max-density bin extrudes to radius 1+boost

export function createDirectionHistogram(canvas) {
  const w = canvas.clientWidth || 220;
  const h = canvas.clientHeight || 220;

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(devicePixelRatio);
  renderer.setSize(w, h, false);

  const scene = new THREE.Scene();
  // FOV/distance picked so the fully-extruded sphere fits with margin.
  const camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 20);
  camera.position.set(0, 0, 4.5);
  camera.lookAt(0, 0, 0);

  scene.add(new THREE.AmbientLight(0xffffff, 0.85));
  const dl = new THREE.DirectionalLight(0xffffff, 0.35);
  dl.position.set(2, 3, 4); scene.add(dl);

  const geom = new THREE.IcosahedronGeometry(1, HIST_SUBDIV);
  // PolyhedronGeometry produces a non-indexed BufferGeometry: one triplet of
  // vertices per face. Per-face coloring is "write the same RGB to vertices
  // [3f, 3f+1, 3f+2]"; per-face radial lift is "scale those 3 vertices by h".
  const posAttr = geom.attributes.position;
  const faceCount = posAttr.count / 3;

  // Centroid unit-vector for each face — used to bin a query direction by
  // argmax of the dot product (the closest face is the densest bin).
  const faceNormals = new Float32Array(faceCount * 3);
  for (let f = 0; f < faceCount; f++) {
    const i = f * 9;
    const cx = (posAttr.array[i  ] + posAttr.array[i+3] + posAttr.array[i+6]) / 3;
    const cy = (posAttr.array[i+1] + posAttr.array[i+4] + posAttr.array[i+7]) / 3;
    const cz = (posAttr.array[i+2] + posAttr.array[i+5] + posAttr.array[i+8]) / 3;
    const inv = 1 / Math.hypot(cx, cy, cz);
    faceNormals[3*f]     = cx * inv;
    faceNormals[3*f + 1] = cy * inv;
    faceNormals[3*f + 2] = cz * inv;
  }
  // Precomputed (θ, φ) → face-index lookup. With it, binning a direction
  // becomes O(1) (atan2 + acos + one array read) instead of 1280 dot
  // products. The brute-force version is the dominant cost at large N
  // (≈90 ms/frame at N=40 in node); the LUT brings the per-frame binning
  // cost down by ~50×, so step + visualisation become the bottleneck again.
  // 512×256 grid means each LUT cell spans ~0.7° on the sphere — much
  // finer than the 1280 face cells (~1.7° linear), so misclassification
  // only happens within a fraction of a face boundary and is visually
  // indistinguishable from the brute-force result. Build cost ≈ 170 M ops,
  // a one-time ~200 ms hit at page load.
  const LUT_T = 512;
  const LUT_P = 256;
  const dirLUT = new Uint16Array(LUT_T * LUT_P);
  {
    const dT = (2 * Math.PI) / LUT_T;
    const dP = Math.PI / LUT_P;
    for (let pi = 0; pi < LUT_P; pi++) {
      const phi = (pi + 0.5) * dP;
      const sphi = Math.sin(phi);
      const cphi = Math.cos(phi);
      for (let ti = 0; ti < LUT_T; ti++) {
        const theta = (ti + 0.5) * dT - Math.PI;
        const x = sphi * Math.cos(theta);
        const y = sphi * Math.sin(theta);
        const z = cphi;
        let bestF = 0, bestDot = -2;
        for (let f = 0; f < faceCount; f++) {
          const fi = f * 3;
          const d = x*faceNormals[fi] + y*faceNormals[fi+1] + z*faceNormals[fi+2];
          if (d > bestDot) { bestDot = d; bestF = f; }
        }
        dirLUT[pi * LUT_T + ti] = bestF;
      }
    }
  }
  const LUT_P_OVER_PI = LUT_P / Math.PI;
  const LUT_T_OVER_2PI = LUT_T / (2 * Math.PI);
  const colors = new Float32Array(posAttr.count * 3);
  geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  const counts = new Uint32Array(faceCount);
  const faceHeights = new Float32Array(faceCount);
  // Snapshot the unit-sphere positions so each frame's lift is derived from
  // scratch (lifted = unit · (1 + boost·t)) rather than accumulated.
  const origPos = posAttr.array.slice();

  const sphereMesh = new THREE.Mesh(geom, new THREE.MeshStandardMaterial({
    vertexColors: true, roughness: 0.65, metalness: 0.0, flatShading: true,
  }));
  sphereMesh.frustumCulled = false;     // bounding sphere = unit; lifts go past it
  scene.add(sphereMesh);

  // Subtle wire overlay so adjacent bins stay distinguishable when colors are
  // similar. Shares the geometry → wires lift with their faces, drawing a
  // dark outline on top of every plateau.
  const wireMesh = new THREE.Mesh(geom, new THREE.MeshBasicMaterial({
    color: 0x000000, wireframe: true, transparent: true, opacity: 0.18,
  }));
  wireMesh.frustumCulled = false;
  scene.add(wireMesh);

  // Walls bridge the height step between every pair of adjacent bins. Need
  // vertex deduplication + face adjacency: PolyhedronGeometry duplicates
  // shared vertices per face, so we collapse them by position-hash and then
  // walk every face's 3 edges to find the two adjacent faces of each edge.
  const dedup = _dedupVertices(posAttr.array);
  const unitDirs = dedup.uniqueDirs;
  const wallEdges = _buildEdges(dedup.faceVerts, faceCount);
  const edgeCount = wallEdges.length / 4;
  // 6 verts/edge (two triangles forming a quad), 3 floats/vert.
  const wallPos = new Float32Array(edgeCount * 18);
  const wallCol = new Float32Array(edgeCount * 18);
  const wallGeom = new THREE.BufferGeometry();
  wallGeom.setAttribute('position', new THREE.BufferAttribute(wallPos, 3));
  wallGeom.setAttribute('color', new THREE.BufferAttribute(wallCol, 3));
  // DoubleSide so the walls are visible from any orbit direction.
  const wallMesh = new THREE.Mesh(wallGeom, new THREE.MeshStandardMaterial({
    vertexColors: true, side: THREE.DoubleSide,
    roughness: 0.7, metalness: 0.0, flatShading: true,
  }));
  wallMesh.frustumCulled = false;
  scene.add(wallMesh);

  const tmpColor = new THREE.Color();

  return {
    update(dirs, count) {
      for (let f = 0; f < faceCount; f++) counts[f] = 0;
      // O(1) per direction via the precomputed LUT.
      for (let bi = 0; bi < count; bi++) {
        const nx = dirs[3*bi], ny = dirs[3*bi + 1], nz = dirs[3*bi + 2];
        let z = nz;
        if (z >  1) z =  1;
        else if (z < -1) z = -1;
        const phi = Math.acos(z);
        const theta = Math.atan2(ny, nx);
        let pi = (phi * LUT_P_OVER_PI) | 0;
        if (pi >= LUT_P) pi = LUT_P - 1;
        let ti = ((theta + Math.PI) * LUT_T_OVER_2PI) | 0;
        if (ti >= LUT_T) ti = LUT_T - 1;
        counts[dirLUT[pi * LUT_T + ti]]++;
      }
      // Re-normalize each frame (fallback maxCount=1 keeps an empty histogram
      // from divide-by-zero painting every face NaN).
      let maxCount = 1;
      for (let f = 0; f < faceCount; f++) {
        if (counts[f] > maxCount) maxCount = counts[f];
      }
      const inv = 1 / maxCount;
      const posArr = geom.attributes.position.array;
      for (let f = 0; f < faceCount; f++) {
        const t = counts[f] * inv;
        // Hue blue (0.66) → red (0.0) as density rises; lightness ramps up
        // so empty bins read as dark and dense bins as bright.
        tmpColor.setHSL(0.66 * (1 - t), 0.85, 0.18 + 0.42 * t);
        const r = tmpColor.r, g = tmpColor.g, bl = tmpColor.b;
        const ci = f * 9;
        colors[ci    ] = r; colors[ci + 1] = g; colors[ci + 2] = bl;
        colors[ci + 3] = r; colors[ci + 4] = g; colors[ci + 5] = bl;
        colors[ci + 6] = r; colors[ci + 7] = g; colors[ci + 8] = bl;
        // Radial extrusion: scale the 3 vertices uniformly outward. Same
        // factor for all 3 keeps the face flat (translated, not deformed).
        const hh = 1 + HIST_RADIUS_BOOST * t;
        faceHeights[f] = hh;
        posArr[ci    ] = origPos[ci    ] * hh;
        posArr[ci + 1] = origPos[ci + 1] * hh;
        posArr[ci + 2] = origPos[ci + 2] * hh;
        posArr[ci + 3] = origPos[ci + 3] * hh;
        posArr[ci + 4] = origPos[ci + 4] * hh;
        posArr[ci + 5] = origPos[ci + 5] * hh;
        posArr[ci + 6] = origPos[ci + 6] * hh;
        posArr[ci + 7] = origPos[ci + 7] * hh;
        posArr[ci + 8] = origPos[ci + 8] * hh;
      }
      geom.attributes.color.needsUpdate = true;
      geom.attributes.position.needsUpdate = true;

      // Walls: for each shared edge, build a quad spanning between the lower
      // and higher of the two adjacent bins, colored to match the higher bin
      // (standard 3D-bar-chart look — a tall red bin extends down as a solid
      // red column to its shorter neighbour).
      for (let e = 0; e < edgeCount; e++) {
        const ei = e * 4;
        const va = wallEdges[ei], vb = wallEdges[ei + 1];
        const f1 = wallEdges[ei + 2], f2 = wallEdges[ei + 3];
        const h1 = faceHeights[f1], h2 = faceHeights[f2];
        const hiF = h1 >= h2 ? f1 : f2;
        const loH = h1 < h2 ? h1 : h2;
        const hiH = h1 > h2 ? h1 : h2;
        const ax = unitDirs[3*va],     ay = unitDirs[3*va + 1], az = unitDirs[3*va + 2];
        const bx = unitDirs[3*vb],     by = unitDirs[3*vb + 1], bz = unitDirs[3*vb + 2];
        const wi = e * 18;
        wallPos[wi     ] = ax * loH; wallPos[wi +  1] = ay * loH; wallPos[wi +  2] = az * loH;
        wallPos[wi +  3] = bx * loH; wallPos[wi +  4] = by * loH; wallPos[wi +  5] = bz * loH;
        wallPos[wi +  6] = ax * hiH; wallPos[wi +  7] = ay * hiH; wallPos[wi +  8] = az * hiH;
        wallPos[wi +  9] = bx * loH; wallPos[wi + 10] = by * loH; wallPos[wi + 11] = bz * loH;
        wallPos[wi + 12] = bx * hiH; wallPos[wi + 13] = by * hiH; wallPos[wi + 14] = bz * hiH;
        wallPos[wi + 15] = ax * hiH; wallPos[wi + 16] = ay * hiH; wallPos[wi + 17] = az * hiH;
        const fci = hiF * 9;
        const r = colors[fci], g = colors[fci + 1], bl = colors[fci + 2];
        for (let v = 0; v < 6; v++) {
          const wci = wi + v * 3;
          wallCol[wci] = r; wallCol[wci + 1] = g; wallCol[wci + 2] = bl;
        }
      }
      wallGeom.attributes.position.needsUpdate = true;
      wallGeom.attributes.color.needsUpdate = true;
    },

    // Match the histogram view direction to the main scene's camera so a
    // face on the histogram visually corresponds to the same world direction.
    syncCamera(mainCamera) {
      const len = Math.hypot(mainCamera.position.x, mainCamera.position.y, mainCamera.position.z);
      if (len < 1e-6) return;
      const s = 4.5 / len;
      camera.position.set(mainCamera.position.x * s,
                          mainCamera.position.y * s,
                          mainCamera.position.z * s);
      camera.up.copy(mainCamera.up);
      camera.lookAt(0, 0, 0);
    },

    render() {
      renderer.render(scene, camera);
    },
  };
}

// Polar (S^1) direction histogram. Bins angles ∈ [0, 2π) into NUM_BINS
// wedges and draws each as an arc whose radial extent grows with the bin
// count. Same external shape as createDirectionHistogram (update / render
// / syncCamera) so the call sites can swap between them by reference.
//
// Hue per bin = bin center angle / 2π so the colors line up with the
// "domains" voxel coloring (which also uses θ → hue in S^1 mode).
export function createAngleHistogram(canvas) {
  const NUM_BINS = 60;
  const counts = new Uint32Array(NUM_BINS);
  const ctx = canvas.getContext('2d');
  // Match the canvas backing-store to its CSS size for crisp rendering.
  function resize() {
    canvas.width  = canvas.clientWidth  || 220;
    canvas.height = canvas.clientHeight || 220;
  }
  resize();

  return {
    update(angles, count) {
      counts.fill(0);
      const inv = NUM_BINS / (2 * Math.PI);
      for (let i = 0; i < count; i++) {
        let theta = angles[i];
        // Wrap negatives or values just above 2π that drift in from noise.
        theta -= 2 * Math.PI * Math.floor(theta / (2 * Math.PI));
        let b = (theta * inv) | 0;
        if (b >= NUM_BINS) b = NUM_BINS - 1;
        counts[b]++;
      }
    },
    syncCamera() { /* no-op: 2D plot has no camera */ },
    render() {
      const w = canvas.width, h = canvas.height;
      ctx.fillStyle = '#0a0d12';
      ctx.fillRect(0, 0, w, h);
      const cx = w / 2, cy = h / 2;
      const rInner = Math.min(w, h) * 0.18;
      const rOuter = Math.min(w, h) * 0.46;
      const dr = rOuter - rInner;
      let maxCount = 1;
      for (let b = 0; b < NUM_BINS; b++) if (counts[b] > maxCount) maxCount = counts[b];
      const inv = 1 / maxCount;
      // The wedges. Canvas 2D angles are CW from +x; we flip y so +θ goes
      // CCW from +x (standard math convention).
      ctx.save();
      ctx.translate(cx, cy);
      ctx.scale(1, -1);
      for (let b = 0; b < NUM_BINS; b++) {
        const a0 = (b / NUM_BINS) * 2 * Math.PI;
        const a1 = ((b + 1) / NUM_BINS) * 2 * Math.PI;
        const t = counts[b] * inv;
        const r1 = rInner + dr * t;
        const lum = 0.18 + 0.42 * t;
        ctx.fillStyle = `hsl(${(b + 0.5) / NUM_BINS * 360}, 85%, ${lum * 100}%)`;
        ctx.beginPath();
        ctx.arc(0, 0, r1,     a0, a1);
        ctx.arc(0, 0, rInner, a1, a0, true);
        ctx.closePath();
        ctx.fill();
      }
      ctx.strokeStyle = '#2a3543';
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(0, 0, rOuter, 0, 2 * Math.PI); ctx.stroke();
      ctx.beginPath(); ctx.arc(0, 0, rInner, 0, 2 * Math.PI); ctx.stroke();
      ctx.restore();
      // Cardinal labels (drawn after restore so text isn't y-flipped).
      ctx.fillStyle = '#7a8696';
      ctx.font = '10px ui-sans-serif, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('0',    cx + rOuter + 10, cy);
      ctx.fillText('π/2',  cx, cy - rOuter - 10);
      ctx.fillText('π',    cx - rOuter - 10, cy);
      ctx.fillText('3π/2', cx, cy + rOuter + 10);
    },
  };
}

// Dedup vertices of a non-indexed BufferGeometry by exact-coordinate hash
// (rounded to 5 decimals to swallow subdivision float wobble).
function _dedupVertices(positions) {
  const map = new Map();
  const dirs = [];
  const faceVerts = new Uint16Array(positions.length / 3);
  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i], y = positions[i + 1], z = positions[i + 2];
    const key = x.toFixed(5) + ',' + y.toFixed(5) + ',' + z.toFixed(5);
    let idx = map.get(key);
    if (idx === undefined) {
      idx = dirs.length / 3;
      dirs.push(x, y, z);
      map.set(key, idx);
    }
    faceVerts[i / 3] = idx;
  }
  return { uniqueDirs: new Float32Array(dirs), faceVerts };
}

// For each undirected edge {vA, vB} shared by two faces in a closed mesh,
// emit [vA, vB, f1, f2]. Closed icosahedron → every edge has exactly 2 faces.
function _buildEdges(faceVerts, faceCount) {
  const edgeMap = new Map();
  for (let f = 0; f < faceCount; f++) {
    const v0 = faceVerts[3*f], v1 = faceVerts[3*f + 1], v2 = faceVerts[3*f + 2];
    const pairs = [[v0, v1], [v1, v2], [v2, v0]];
    for (const [a, b] of pairs) {
      const lo = a < b ? a : b, hi = a < b ? b : a;
      const key = lo * 100000 + hi;
      const existing = edgeMap.get(key);
      if (existing === undefined) edgeMap.set(key, [f, -1]);
      else existing[1] = f;
    }
  }
  const out = [];
  for (const [key, [f1, f2]] of edgeMap) {
    if (f2 === -1) continue;          // boundary edge — shouldn't happen here
    const lo = (key / 100000) | 0, hi = key % 100000;
    out.push(lo, hi, f1, f2);
  }
  return new Uint32Array(out);
}
