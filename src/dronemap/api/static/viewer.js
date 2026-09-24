/* ============================================================
   DroneMap Workstation — 3D Viewport Engine (Three.js)
   ============================================================ */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

console.log('[dronemap] Three.js Engine initialized');

/* Report an engine failure to the user instead of dying quietly.
   Everything below used to sit inside a bare `if (canvas) {...}` with no error
   handling, so a WebGL refusal or a missing vendor file produced exactly the
   same screen as "no project selected": a valid model on disk, and a UI that
   never admitted anything was wrong. */
function reportEngineFailure(detail) {
  console.error('[dronemap] viewer engine failure:', detail);
  if (window.__dronemapShowViewerFailure) {
    window.__dronemapShowViewerFailure(detail);
    return;
  }
  const overlay = document.getElementById('status-overlay');
  if (overlay) {
    overlay.style.display = 'block';
    overlay.innerHTML =
      '<div class="status-card"><div class="status-title" style="color:#f87171">' +
      '3D viewer failed to start</div><div class="status-desc">' + detail + '</div></div>';
  }
}

// Three.js Scene Setup
const canvas = document.getElementById('three-canvas');
if (!canvas) {
  reportEngineFailure('The 3D canvas element (#three-canvas) is missing from the page.');
} else try {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.LinearToneMapping;
  renderer.toneMappingExposure = 1.25;

  const scene = new THREE.Scene();
  // Clean neutral dark background
  scene.background = new THREE.Color(0x0e131d);

  const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 10000);
  camera.position.set(0, 100, 160);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.maxDistance = 6000;
  controls.minDistance = 0.5;

  // Balanced Photogrammetric Sun & Fill Lighting
  const ambient = new THREE.AmbientLight(0xffffff, 1.2);
  scene.add(ambient);

  const hemi = new THREE.HemisphereLight(0xffffff, 0x475569, 1.0);
  scene.add(hemi);

  const sun1 = new THREE.DirectionalLight(0xffffff, 1.1);
  sun1.position.set(120, 220, 100);
  scene.add(sun1);

  const sun2 = new THREE.DirectionalLight(0xffffff, 0.8);
  sun2.position.set(-120, 160, -100);
  scene.add(sun2);

  const underFill = new THREE.DirectionalLight(0xffffff, 0.5);
  underFill.position.set(0, -100, 0);
  scene.add(underFill);

  // Ground Spatial Reference Grid
  const grid = new THREE.GridHelper(600, 60, 0x334155, 0x1e293b);
  grid.position.y = -0.1;
  scene.add(grid);

  // Resize Handler
  //
  // A single call at module time is not enough: modules evaluate before
  // DOMContentLoaded, so the flex layout may not have been measured yet. A
  // zero-size renderer draws nothing, no `resize` event ever follows, and the
  // viewport stays permanently black -- not even the reference grid appears,
  // which is exactly the symptom of "the model will not display".
  function resize() {
    const vp = document.getElementById('viewport');
    if (!vp) return false;
    const w = vp.clientWidth;
    const h = vp.clientHeight;
    if (w < 2 || h < 2) return false;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    return true;
  }
  window.addEventListener('resize', resize);

  if (!resize()) {
    // Layout not settled yet. Retry on the next frame, and keep a
    // ResizeObserver so a sidebar toggle or window change is always tracked.
    requestAnimationFrame(resize);
  }
  if (typeof ResizeObserver !== 'undefined') {
    const vp = document.getElementById('viewport');
    if (vp) new ResizeObserver(resize).observe(vp);
  }

  // Animation Render Loop
  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  // State
  let currentModel = null;
  let measureMode = false;
  let measurePoints = [];
  let activeShadingMode = 'textured';
  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2();

  const statusOverlay     = document.getElementById('status-overlay');
  const activeProjectName = document.getElementById('active-project-name');
  const measurePill       = document.getElementById('measure-pill');
  const measurePillText   = document.getElementById('measure-pill-text');
  const measureResultCard = document.getElementById('measure-result');

  const gltfLoader = new GLTFLoader();

  // Shading Controller
  function applyShadingMode(mode) {
    activeShadingMode = mode;
    const btnTex = document.getElementById('btn-shade-textured');
    const btnWhite = document.getElementById('btn-shade-white');
    if (btnTex) btnTex.classList.toggle('active', mode === 'textured');
    if (btnWhite) btnWhite.classList.toggle('active', mode === 'white');

    if (!currentModel) return;

    currentModel.traverse((child) => {
      // Only mesh nodes have a .material. Group, Object3D, Bone, etc. do not.
      // Accessing .material on them throws TypeError and aborts the traverse,
      // leaving the mesh with no shading applied (renders black).
      if (!child.isMesh || !child.material) return;

      if (mode === 'white') {
        child.material.map = null;
        child.material.vertexColors = false;
        child.material.color.setHex(0xe2e8f0);
        child.material.roughness = 0.55;
        child.material.metalness = 0.04;
      } else if (mode === 'textured') {
        if (child.userData.originalMap) {
          child.material.map = child.userData.originalMap;
        }
        if (child.userData.hasVertexColors) {
          child.material.vertexColors = true;
        }
        if (child.userData.originalColor) {
          child.material.color.copy(child.userData.originalColor);
        }
        child.material.roughness = 0.9;
        child.material.metalness = 0.0;
      }
      child.material.needsUpdate = true;
    });
  }

  // Model Loading
  //
  // ``variant`` selects which texture atlas to show. A run whose seam-levelled
  // atlas came out black is re-textured without levelling and keeps both, so
  // the artefact (a dark mesh crossed by coloured lines) can be compared
  // directly against the corrected one rather than described.
  window.load3DModel = function(runId, variant) {
    variant = variant === 'alt' ? 'alt' : 'primary';
    window.currentVariant = variant;
    const assetUrl = variant === 'alt'
      ? `/api/runs/${runId}/model_alt.glb`
      : `/api/runs/${runId}/model.glb`;

    const btnPrimary = document.getElementById('btn-variant-primary');
    const btnAlt = document.getElementById('btn-variant-alt');
    if (btnPrimary) btnPrimary.classList.toggle('active', variant === 'primary');
    if (btnAlt) btnAlt.classList.toggle('active', variant === 'alt');

    if (currentModel) {
      scene.remove(currentModel);
      currentModel = null;
    }
    window.currentRunId = runId;
    if (activeProjectName) activeProjectName.textContent = runId;
    if (measureResultCard) measureResultCard.style.display = 'none';

    if (statusOverlay) {
      statusOverlay.style.display = 'block';
      statusOverlay.innerHTML = `
        <div class="status-card">
          <div class="pulsing-spinner" style="margin: 0 auto 12px;"></div>
          <div class="status-title">Loading 3D Model…</div>
          <div class="status-desc">Reading geometry and texture data for <strong>${runId}</strong></div>
        </div>
      `;
    }

    gltfLoader.load(
      assetUrl,
      (gltf) => {
        currentModel = gltf.scene;

        // Center model over origin and position baseline on ground grid (Y=0)
        currentModel.updateMatrixWorld(true);

        const box = new THREE.Box3().setFromObject(currentModel);
        const centre = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        currentModel.position.x -= centre.x;
        currentModel.position.z -= centre.z;
        currentModel.position.y -= box.min.y;

        const maxAniso = renderer.capabilities.getMaxAnisotropy();
        currentModel.traverse((child) => {
          if (child.isMesh) {
            // OpenMVS GLB exports often omit the NORMAL accessor entirely (the
            // JSON shows only POSITION + TEXCOORD_0). THREE.js does not auto-
            // compute normals in that case — we must trigger it explicitly.
            if (child.geometry && !child.geometry.attributes.normal) {
              child.geometry.computeVertexNormals();
            }
            if (child.material) {
              child.material.side = THREE.DoubleSide;
              child.material.shadowSide = THREE.DoubleSide;
              child.material.roughness = 0.9;
              child.material.metalness = 0.0;
              if (child.geometry && child.geometry.attributes.color) {
                child.material.vertexColors = true;
                child.userData.hasVertexColors = true;
              }
              if (child.material.map) {
                child.material.map.colorSpace = THREE.SRGBColorSpace;
                child.material.map.anisotropy = maxAniso;
                child.material.map.generateMipmaps = true;
                child.material.map.minFilter = THREE.LinearMipmapLinearFilter;
                child.material.map.magFilter = THREE.LinearFilter;
                child.material.map.needsUpdate = true;
                child.userData.originalMap = child.material.map;
              }
              child.userData.originalColor = child.material.color ? child.material.color.clone() : new THREE.Color(0xffffff);
              child.material.needsUpdate = true;
            }
            child.castShadow = true;
            child.receiveShadow = true;
          }
        });

        applyShadingMode(activeShadingMode);
        scene.add(currentModel);

        const maxDim = Math.max(size.x, size.y, size.z, 1.0);
        camera.near = Math.max(0.01, maxDim / 1000);
        camera.far = Math.max(10000, maxDim * 50);
        camera.updateProjectionMatrix();

        camera.position.set(0, maxDim * 0.85, maxDim * 1.35);
        controls.target.set(0, size.y * 0.35, 0);
        controls.maxDistance = maxDim * 20;
        controls.minDistance = maxDim * 0.01;
        controls.update();

        if (statusOverlay) statusOverlay.style.display = 'none';
      },
      (xhr) => {
        const pct = xhr.total ? Math.round((xhr.loaded / xhr.total) * 100) : '';
        if (pct && statusOverlay) {
          const desc = statusOverlay.querySelector('.status-desc');
          if (desc) desc.textContent = `Streaming asset… ${pct}%`;
        }
      },
      (err) => {
        if (statusOverlay) {
          statusOverlay.innerHTML = `
            <div class="status-card">
              <div class="status-title" style="color:var(--danger)">Model Not Available</div>
              <div class="status-desc">${err.message || '3D geometry file (model.glb) not found for this run.'}</div>
            </div>
          `;
        }
      }
    );
  };

  // 3D Metric Measurement
  function setMeasureMode(on) {
    measureMode = on;
    const btnM = document.getElementById('btn-measure');
    const btnO = document.getElementById('btn-orbit');
    if (btnM) btnM.classList.toggle('active', on);
    if (btnO) btnO.classList.toggle('active', !on);
    controls.enabled = !on;
    if (measurePill) measurePill.style.display = on ? 'flex' : 'none';
    if (!on) clearMeasure();
  }

  let activeMarkers = [];
  function clearMeasure() {
    measurePoints.forEach(p => scene.remove(p.marker));
    measurePoints = [];
    activeMarkers.forEach(m => scene.remove(m));
    activeMarkers = [];
    if (measureLine) scene.remove(measureLine);
    measureLine = null;
    if (measureResultCard) measureResultCard.style.display = 'none';
    if (measurePillText) measurePillText.textContent = 'Click two surface points to compute distance';
  }

  function addMarker(pos) {
    const geo = new THREE.SphereGeometry(0.4, 16, 16);
    const mat = new THREE.MeshBasicMaterial({ color: 0x0284c7 });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.copy(pos);
    scene.add(mesh);
    activeMarkers.push(mesh);
    return mesh;
  }

  let measureLine = null;
  function drawLine(a, b) {
    if (measureLine) scene.remove(measureLine);
    const geo = new THREE.BufferGeometry().setFromPoints([a, b]);
    const mat = new THREE.LineDashedMaterial({ color: 0x38bdf8, dashSize: 1.2, gapSize: 0.6, linewidth: 2 });
    measureLine = new THREE.Line(geo, mat);
    measureLine.computeLineDistances();
    scene.add(measureLine);
  }

  canvas.addEventListener('click', (e) => {
    if (!measureMode || !currentModel) return;

    const rect = canvas.getBoundingClientRect();
    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObject(currentModel, true);
    if (!hits.length) return;

    const hit = hits[0].point;
    const marker = addMarker(hit);
    measurePoints.push({ pos: hit.clone(), marker });

    if (measurePoints.length === 1) {
      if (measurePillText) measurePillText.textContent = 'Point 1 selected. Click point 2…';
    } else if (measurePoints.length === 2) {
      const a = measurePoints[0].pos;
      const b = measurePoints[1].pos;
      drawLine(a, b);

      if (window.currentRunId) {
        fetch(`/api/runs/${window.currentRunId}/measure`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ a: [a.x, a.y, a.z], b: [b.x, b.y, b.z] }),
        })
          .then(r => r.json())
          .then(data => {
            if (data.is_metric) {
              document.getElementById('val-3d-dist').textContent = `${data.distance_m.toFixed(2)} m`;
              document.getElementById('val-horiz-dist').textContent = `${data.horizontal_m.toFixed(2)} m`;
              document.getElementById('val-vert-dist').textContent = `${data.vertical_m.toFixed(2)} m`;
              document.getElementById('val-bearing').textContent = `${data.bearing_deg.toFixed(1)}°`;
              if (measurePillText) measurePillText.textContent = `Distance: ${data.distance_m.toFixed(2)} m (Click again for new measurement)`;
            } else {
              document.getElementById('val-3d-dist').textContent = `${data.distance_m.toFixed(2)} (relative units)`;
              document.getElementById('val-horiz-dist').textContent = '— (unscaled)';
              document.getElementById('val-vert-dist').textContent = '— (unscaled)';
              document.getElementById('val-bearing').textContent = '— (unoriented)';
              if (measurePillText) measurePillText.textContent = `Distance: ${data.distance_m.toFixed(2)} rel units (No GPS scale)`;
            }
            if (measureResultCard) measureResultCard.style.display = 'block';
          })
          .catch(() => {
            const d = a.distanceTo(b);
            document.getElementById('val-3d-dist').textContent = `${d.toFixed(2)} (rel)`;
            document.getElementById('val-horiz-dist').textContent = '—';
            document.getElementById('val-vert-dist').textContent = '—';
            document.getElementById('val-bearing').textContent = '—';
            if (measureResultCard) measureResultCard.style.display = 'block';
          });
      }

      measurePoints = [];
    }
  });

  // Toolbar Button Connections
  const btnOrbit = document.getElementById('btn-orbit');
  const btnMeasure = document.getElementById('btn-measure');
  const btnCloseMeasure = document.getElementById('btn-close-measure');
  const btnReset = document.getElementById('btn-reset');
  const chkWireframe = document.getElementById('chk-wireframe');
  const chkAutorotate = document.getElementById('chk-autorotate');

  const btnShadeTextured = document.getElementById('btn-shade-textured');
  const btnShadeWhite = document.getElementById('btn-shade-white');

  if (btnShadeTextured) {
    btnShadeTextured.addEventListener('click', () => applyShadingMode('textured'));
  }
  if (btnShadeWhite) {
    btnShadeWhite.addEventListener('click', () => applyShadingMode('white'));
  }

  const btnVariantPrimary = document.getElementById('btn-variant-primary');
  const btnVariantAlt = document.getElementById('btn-variant-alt');
  if (btnVariantPrimary) {
    btnVariantPrimary.addEventListener('click', () => {
      if (window.currentRunId) window.load3DModel(window.currentRunId, 'primary');
    });
  }
  if (btnVariantAlt) {
    btnVariantAlt.addEventListener('click', () => {
      if (window.currentRunId) window.load3DModel(window.currentRunId, 'alt');
    });
  }

  if (btnOrbit) btnOrbit.addEventListener('click', () => setMeasureMode(false));
  if (btnMeasure) btnMeasure.addEventListener('click', () => setMeasureMode(!measureMode));
  if (btnCloseMeasure) btnCloseMeasure.addEventListener('click', clearMeasure);

  if (btnReset) {
    btnReset.addEventListener('click', () => {
      if (!currentModel) return;
      const box = new THREE.Box3().setFromObject(currentModel);
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);
      camera.position.set(0, maxDim * 0.8, maxDim * 1.3);
      controls.target.set(0, 0, 0);
      controls.update();
    });
  }

  if (chkWireframe) {
    chkWireframe.addEventListener('change', (e) => {
      if (!currentModel) return;
      currentModel.traverse((child) => {
        if (child.isMesh && child.material) {
          child.material.wireframe = e.target.checked;
        }
      });
    });
  }

  if (chkAutorotate) {
    chkAutorotate.addEventListener('change', (e) => {
      controls.autoRotate = e.target.checked;
      controls.autoRotateSpeed = 1.2;
    });
  }

  // AI Copilot 3D Spatial Actions
  window.triggerViewerAction = (action) => {
    if (!action || !currentModel) return;
    if (action.type === 'highlight_confidence') {
      currentModel.traverse((child) => {
        if (child.isMesh && child.material) {
          const origColor = child.material.color ? child.material.color.clone() : new THREE.Color(0xffffff);
          child.material.color = new THREE.Color(0x3fb950);
          setTimeout(() => {
            if (child.material) child.material.color = origColor;
          }, 2500);
        }
      });
    } else if (action.type === 'toggle_masks') {
      currentModel.traverse((child) => {
        if (child.isMesh && child.material) {
          const origColor = child.material.color ? child.material.color.clone() : new THREE.Color(0xffffff);
          child.material.color = new THREE.Color(0xf85149);
          setTimeout(() => {
            if (child.material) child.material.color = origColor;
          }, 2500);
        }
      });
    }
  };

  // Notify UI controller that viewer engine is ready
  window.viewerReady = true;
  window.dispatchEvent(new CustomEvent('viewer-ready'));
  if (window.pendingRunId) {
    console.log('[dronemap] viewer initialized; loading pending run:', window.pendingRunId);
    window.load3DModel(window.pendingRunId);
    window.pendingRunId = null;
  }
} catch (err) {
  // Anything thrown above (WebGL context refused, shader compile failure, a
  // vendored Three.js file that 404'd) is surfaced rather than swallowed.
  reportEngineFailure(
    (err && err.message) ? err.message : 'Unknown error while initialising the 3D engine.'
  );
}
