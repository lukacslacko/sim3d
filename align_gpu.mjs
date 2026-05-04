// GPU implementation of the orientation-alignment iteration.
//
// Layout: all N³ quaternions live in a single 2D RGBA32F render target,
// width = N and height = N·N — z-slices stack vertically, so texel
// (ix, iy + iz·N) holds cell (ix, iy, iz)'s quaternion. Pixel index in
// readback then equals the cell index `(iz·N + iy)·N + ix` used by the CPU
// side, so we can blit straight into state.quats with no reordering.
//
// Each step is one full-screen fragment-shader pass: read self + 6 PBC
// neighbors, sign-fix each into the same hemisphere as self, normalize the
// sum, lerp by η, optional hash-based noise, renormalize. Two render
// targets ping-pong so reads-from-old / writes-to-new stay symmetric.
//
// stepsPerFrame N steps run entirely on GPU; one readPixels at the end
// brings the field back to CPU for the visualization. Requires WebGL2 +
// EXT_color_buffer_float — both are auto-enabled by three.js when a
// FloatType render target is created. Returns null on WebGL1.

import * as THREE from 'three';

export function createGPUAligner(renderer, N) {
  if (!renderer.capabilities.isWebGL2) return null;

  const w = N;
  const h = N * N;

  const targetOpts = {
    type: THREE.FloatType,
    format: THREE.RGBAFormat,
    minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter,
    depthBuffer: false,
    stencilBuffer: false,
    generateMipmaps: false,
  };

  let read  = new THREE.WebGLRenderTarget(w, h, targetOpts);
  let write = new THREE.WebGLRenderTarget(w, h, targetOpts);

  const quadGeom = new THREE.PlaneGeometry(2, 2);
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

  // Compute material: one alignment step per draw.
  const stepMat = new THREE.ShaderMaterial({
    uniforms: {
      tQuat:   { value: null },
      u_eta:   { value: 0.2 },
      u_noise: { value: 0.0 },
      u_seed:  { value: 0.0 },
      u_N:     { value: N },
    },
    vertexShader: `
      void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }
    `,
    fragmentShader: `
      precision highp float;
      precision highp int;
      uniform sampler2D tQuat;
      uniform float u_eta;
      uniform float u_noise;
      uniform float u_seed;
      uniform int u_N;

      vec4 fetchQuat(int ix, int iy, int iz) {
        // PBC wrap on each axis.
        int N = u_N;
        ix = ((ix % N) + N) % N;
        iy = ((iy % N) + N) % N;
        iz = ((iz % N) + N) % N;
        // Texel center in normalized UV.
        vec2 uv = (vec2(float(ix), float(iy + iz * N)) + 0.5)
                / vec2(float(N), float(N * N));
        return texture2D(tQuat, uv);
      }

      // Cheap deterministic hash. Salted with u_seed so successive steps
      // get different noise patterns despite the same cell coords.
      float hash(vec3 p, float salt) {
        return fract(sin(dot(p + vec3(salt + u_seed),
          vec3(12.9898, 78.233, 37.719))) * 43758.5453);
      }

      void main() {
        int N = u_N;
        ivec2 px = ivec2(gl_FragCoord.xy);
        int ix = px.x;
        int iz = px.y / N;
        int iy = px.y - iz * N;

        vec4 q = fetchQuat(ix, iy, iz);

        // Sign-fixed quaternion sum across the 6 face-adjacent neighbors.
        vec4 s = vec4(0.0);
        vec4 nq;
        nq = fetchQuat(ix - 1, iy, iz); if (dot(nq, q) < 0.0) nq = -nq; s += nq;
        nq = fetchQuat(ix + 1, iy, iz); if (dot(nq, q) < 0.0) nq = -nq; s += nq;
        nq = fetchQuat(ix, iy - 1, iz); if (dot(nq, q) < 0.0) nq = -nq; s += nq;
        nq = fetchQuat(ix, iy + 1, iz); if (dot(nq, q) < 0.0) nq = -nq; s += nq;
        nq = fetchQuat(ix, iy, iz - 1); if (dot(nq, q) < 0.0) nq = -nq; s += nq;
        nq = fetchQuat(ix, iy, iz + 1); if (dot(nq, q) < 0.0) nq = -nq; s += nq;

        // Normalize the average; degenerate fall-back leaves self unchanged.
        float len = length(s);
        if (len > 1e-8) s /= len;
        else s = q;

        // Lerp toward avg, then optional per-component noise.
        vec4 nq2 = mix(q, s, u_eta);
        if (u_noise > 0.0) {
          vec3 base = vec3(float(ix), float(iy), float(iz));
          nq2.x += (hash(base, 0.10) * 2.0 - 1.0) * u_noise;
          nq2.y += (hash(base, 0.27) * 2.0 - 1.0) * u_noise;
          nq2.z += (hash(base, 0.51) * 2.0 - 1.0) * u_noise;
          nq2.w += (hash(base, 0.83) * 2.0 - 1.0) * u_noise;
        }
        gl_FragColor = normalize(nq2);
      }
    `,
  });
  const stepScene = new THREE.Scene();
  stepScene.add(new THREE.Mesh(quadGeom, stepMat));

  // Pass-through material for blitting CPU-side data into the read target.
  const passMat = new THREE.ShaderMaterial({
    uniforms: {
      tSrc:   { value: null },
      u_size: { value: new THREE.Vector2(w, h) },
    },
    vertexShader: `
      void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }
    `,
    fragmentShader: `
      precision highp float;
      uniform sampler2D tSrc;
      uniform vec2 u_size;
      void main() {
        gl_FragColor = texture2D(tSrc, gl_FragCoord.xy / u_size);
      }
    `,
  });
  const passScene = new THREE.Scene();
  passScene.add(new THREE.Mesh(quadGeom, passMat));

  // Save / restore the renderer's current target around our internal renders
  // so we don't clobber whatever the caller had set up.
  let stepCounter = 0;

  function renderTo(target, scene) {
    const prev = renderer.getRenderTarget();
    renderer.setRenderTarget(target);
    renderer.render(scene, camera);
    renderer.setRenderTarget(prev);
  }

  return {
    // Copy the CPU buffer into the read target. Float32Array length N³·4,
    // cell-major: same layout as state.quats.
    upload(quats) {
      const src = new THREE.DataTexture(quats, w, h,
        THREE.RGBAFormat, THREE.FloatType);
      src.minFilter = THREE.NearestFilter;
      src.magFilter = THREE.NearestFilter;
      src.needsUpdate = true;
      passMat.uniforms.tSrc.value = src;
      renderTo(read, passScene);
      passMat.uniforms.tSrc.value = null;
      src.dispose();
    },

    step(eta, noise) {
      stepMat.uniforms.tQuat.value = read.texture;
      stepMat.uniforms.u_eta.value = eta;
      stepMat.uniforms.u_noise.value = noise;
      // Cheap evolving seed so successive steps don't reuse the same noise
      // pattern; modulo keeps the float bounded.
      stepMat.uniforms.u_seed.value = ((stepCounter++) % 1024) * 0.7531;
      renderTo(write, stepScene);
      [read, write] = [write, read];
    },

    // Float32Array length N³·4. THREE handles WebGL2 float readPixels
    // (EXT_color_buffer_float). Pixel index = cell index, no reorder.
    readBack(out) {
      renderer.readRenderTargetPixels(read, 0, 0, w, h, out);
    },

    dispose() {
      read.dispose();
      write.dispose();
      stepMat.dispose();
      passMat.dispose();
      quadGeom.dispose();
    },
  };
}
