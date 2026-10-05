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
  renderer.shadowMap.enabled = false;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;

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

  // Balanced Photogrammetric Sun & Fill Lighting - lifts dark shadows without highlight blowout
  const ambient = new THREE.AmbientLight(0xffffff, 0.85);
  scene.add(ambient);

  const hemi = new THREE.HemisphereLight(0xffffff, 0x334455, 0.55);
  scene.add(hemi);

  const sun1 = new THREE.DirectionalLight(0xffffff, 0.75);
  sun1.position.set(120, 220, 100);
  scene.add(sun1);

  const sun2 = new THREE.DirectionalLight(0xffffff, 0.40);
  sun2.position.set(-120, 160, -100);
  scene.add(sun2);

  const underFill = new THREE.DirectionalLight(0xffffff, 0.35);
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
  const chkWireframe      = document.getElementById('chk-wireframe');

  const gltfLoader = new GLTFLoader();

  // Feature Distinction Attribute Generator:
  // Computes per-vertex semantic classification and Turbo elevation colors with direct typed-array access
  function computeDistinctionAttributes(child) {
    const geom = child.geometry;
    if (!geom || !geom.attributes.position) return;
    const pos = geom.attributes.position;
    const count = pos.count;
    const posArr = pos.array;

    if (!geom.attributes.normal) {
      geom.computeVertexNormals();
    }
    const norm = geom.attributes.normal;
    const normArr = norm ? norm.array : null;

    geom.computeBoundingBox();
    const box = geom.boundingBox;
    const minY = box.min.y;
    const maxY = box.max.y;
    const spanX = Math.max(box.max.x - box.min.x, 0.001);
    const spanZ = Math.max(box.max.z - box.min.z, 0.001);

    // Compute robust Y percentile span (p02 to p98) to avoid compression from outlier floaters
    const sampleStep = Math.max(1, Math.floor(count / 1500));
    const sampleY = [];
    for (let i = 0; i < count; i += sampleStep) sampleY.push(posArr[i * 3 + 1]);
    sampleY.sort((a, b) => a - b);
    const p02 = sampleY[Math.floor(sampleY.length * 0.02)] ?? minY;
    const p98 = sampleY[Math.floor(sampleY.length * 0.98)] ?? maxY;
    const robustSpanY = Math.max(p98 - p02, 1.0);

    const heightColors = new Float32Array(count * 3);
    const semanticColors = new Float32Array(count * 3);

    // High-resolution Turbo colormap polynomial approximation
    function turboRGB(t) {
      t = Math.max(0, Math.min(1, t));
      const r = 0.1357 + t * (4.61539 - t * (42.6603 - t * (132.131 - t * (152.55 - t * 59.2863))));
      const g = 0.0914 + t * (2.19418 + t * (4.84296 - t * (14.1850 + t * (4.27726 - t * 2.82956))));
      const b = 0.1067 + t * (12.5833 - t * (60.1974 - t * (109.074 - t * (88.5085 - t * 26.8183))));
      return [Math.max(0, Math.min(1, r)), Math.max(0, Math.min(1, g)), Math.max(0, Math.min(1, b))];
    }

    // Coarse 2D grid to compute local ground datum
    const gridRes = 40;
    const groundCells = Array.from({ length: gridRes * gridRes }, () => []);
    const sampleGridStep = Math.max(1, Math.floor(count / 15000));
    for (let i = 0; i < count; i += sampleGridStep) {
      const idx3 = i * 3;
      const x = posArr[idx3];
      const y = posArr[idx3 + 1];
      const z = posArr[idx3 + 2];
      const gx = Math.max(0, Math.min(gridRes - 1, Math.floor(((x - box.min.x) / spanX) * gridRes)));
      const gz = Math.max(0, Math.min(gridRes - 1, Math.floor(((z - box.min.z) / spanZ) * gridRes)));
      groundCells[gz * gridRes + gx].push(y);
    }

    const groundGrid = new Float32Array(gridRes * gridRes).fill(NaN);
    for (let idx = 0; idx < groundCells.length; idx++) {
      const arr = groundCells[idx];
      if (arr.length > 0) {
        arr.sort((a, b) => a - b);
        groundGrid[idx] = arr[Math.floor(arr.length * 0.15)];
      }
    }

    // Fill missing cells with neighbor mean
    let defaultGround = p02;
    for (let gz = 0; gz < gridRes; gz++) {
      for (let gx = 0; gx < gridRes; gx++) {
        const idx = gz * gridRes + gx;
        if (isNaN(groundGrid[idx])) {
          let closestDist = Infinity;
          let bestVal = defaultGround;
          for (let oz = Math.max(0, gz - 4); oz <= Math.min(gridRes - 1, gz + 4); oz++) {
            for (let ox = Math.max(0, gx - 4); ox <= Math.min(gridRes - 1, gx + 4); ox++) {
              const oIdx = oz * gridRes + ox;
              if (!isNaN(groundGrid[oIdx])) {
                const d = (oz - gz) ** 2 + (ox - gx) ** 2;
                if (d < closestDist) {
                  closestDist = d;
                  bestVal = groundGrid[oIdx];
                }
              }
            }
          }
          groundGrid[idx] = bestVal;
        }
      }
    }

    for (let i = 0; i < count; i++) {
      const idx3 = i * 3;
      const x = posArr[idx3];
      const y = posArr[idx3 + 1];
      const z = posArr[idx3 + 2];
      const nx = normArr ? normArr[idx3] : 0;
      const ny = normArr ? normArr[idx3 + 1] : 1;
      const nz = normArr ? normArr[idx3 + 2] : 0;

      // 1. Height map colors (absolute normalized elevation)
      const normY = (y - p02) / robustSpanY;
      const [hr, hg, hb] = turboRGB(normY);
      heightColors[idx3] = hr;
      heightColors[idx3 + 1] = hg;
      heightColors[idx3 + 2] = hb;

      // 2. Semantic feature distinction (relative height above local ground)
      const gx = Math.max(0, Math.min(gridRes - 1, Math.floor(((x - box.min.x) / spanX) * gridRes)));
      const gz = Math.max(0, Math.min(gridRes - 1, Math.floor(((z - box.min.z) / spanZ) * gridRes)));
      const localBase = groundGrid[gz * gridRes + gx];
      const relHeight = y - (isFinite(localBase) ? localBase : p02);

      const isVertical = Math.abs(ny) < 0.50;
      const isUpward = ny > 0.65;

      let sr, sg, sb;
      if (relHeight < 0.7 && isUpward) {
        // Road / Asphalt: Dark Slate Charcoal (#3b4252)
        sr = 0.23; sg = 0.26; sb = 0.32;
      } else if (relHeight > 2.2 && (isVertical || (isUpward && relHeight > 3.0))) {
        // Houses / Buildings / Roofs: Terracotta Red (#e11d48)
        sr = 0.88; sg = 0.12; sb = 0.28;
      } else if (relHeight > 1.2 && isVertical && (Math.abs(nx) > 0.60 || Math.abs(nz) > 0.60) && relHeight < 4.5) {
        // Poles / Lampposts / Vertical Objects: Electric Cyan (#06b6d4)
        sr = 0.02; sg = 0.71; sb = 0.83;
      } else if (relHeight >= 0.7 && relHeight <= 18.0) {
        // Trees / Vegetation Canopy: Forest Emerald Green (#16a34a)
        sr = 0.09; sg = 0.64; sb = 0.29;
      } else {
        // Terrain / Ground / Bare Earth: Warm Golden Tan (#d97706)
        sr = 0.85; sg = 0.47; sb = 0.02;
      }

      semanticColors[idx3] = sr;
      semanticColors[idx3 + 1] = sg;
      semanticColors[idx3 + 2] = sb;
    }

    geom.userData.heightColors = heightColors;
    geom.userData.semanticColors = semanticColors;
    geom.userData.originalColorArray = geom.attributes.color ? geom.attributes.color.array.slice() : null;
  }

  function updateMeshColors(geom, colorArray) {
    if (!geom) return;
    if (!geom.attributes.color) {
      geom.setAttribute('color', new THREE.BufferAttribute(new Float32Array(geom.attributes.position.count * 3), 3));
    }
    if (colorArray) {
      geom.attributes.color.copyArray(colorArray);
      geom.attributes.color.needsUpdate = true;
    }
  }

  function updateWireframeOverlay(child, isVisible) {
    if (!child.isMesh || !child.material) return;
    const mat = child.userData.currentShadedMaterial || child.userData.originalMaterial || child.material;
    child.material = mat;
    child.material.wireframe = isVisible;
    child.material.needsUpdate = true;
  }

  // Shading Controller
  function applyShadingMode(mode) {
    activeShadingMode = mode;
    const btnTex = document.getElementById('btn-shade-textured');
    const btnSem = document.getElementById('btn-shade-semantic');
    const btnHgt = document.getElementById('btn-shade-height');
    const btnWhite = document.getElementById('btn-shade-white');
    const legSem = document.getElementById('semantic-legend');
    const legHgt = document.getElementById('height-legend');

    if (btnTex) btnTex.classList.toggle('active', mode === 'textured');
    if (btnSem) btnSem.classList.toggle('active', mode === 'semantic');
    if (btnHgt) btnHgt.classList.toggle('active', mode === 'height');
    if (btnWhite) btnWhite.classList.toggle('active', mode === 'white');

    if (legSem) legSem.style.display = mode === 'semantic' ? 'block' : 'none';
    if (legHgt) legHgt.style.display = mode === 'height' ? 'block' : 'none';

    if (!currentModel) return;

    const isWire = chkWireframe ? chkWireframe.checked : false;

    currentModel.traverse((child) => {
      if (!child.isMesh || !child.material) return;
      const geom = child.geometry;
      if (!geom) return;

      if (mode === 'semantic' || mode === 'height') {
        if (!geom.userData.semanticColors || !geom.userData.heightColors) {
          computeDistinctionAttributes(child);
        }
      }

      if (mode === 'white') {
        if (!child.userData.clayMaterial) {
          child.userData.clayMaterial = new THREE.MeshStandardMaterial({
            color: 0xe2e8f0,
            roughness: 0.55,
            metalness: 0.04,
            side: THREE.DoubleSide,
          });
        }
        child.userData.currentShadedMaterial = child.userData.clayMaterial;
      } else if (mode === 'semantic') {
        if (!child.userData.distinctMaterial) {
          child.userData.distinctMaterial = new THREE.MeshStandardMaterial({
            vertexColors: true,
            roughness: 0.70,
            metalness: 0.02,
            side: THREE.DoubleSide,
          });
        }
        updateMeshColors(geom, geom.userData.semanticColors);
        child.userData.currentShadedMaterial = child.userData.distinctMaterial;
      } else if (mode === 'height') {
        if (!child.userData.distinctMaterial) {
          child.userData.distinctMaterial = new THREE.MeshStandardMaterial({
            vertexColors: true,
            roughness: 0.70,
            metalness: 0.02,
            side: THREE.DoubleSide,
          });
        }
        updateMeshColors(geom, geom.userData.heightColors);
        child.userData.currentShadedMaterial = child.userData.distinctMaterial;
      } else if (mode === 'textured') {
        if (geom && geom.userData.originalColorArray) {
          updateMeshColors(geom, geom.userData.originalColorArray);
        } else if (geom && geom.attributes.color) {
          geom.attributes.color.array.fill(1.0);
          geom.attributes.color.needsUpdate = true;
        }
        child.userData.currentShadedMaterial = child.userData.originalMaterial;
      }

      child.material = child.userData.currentShadedMaterial || child.material;
      child.material.wireframe = isWire;
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
    if (!variant) variant = 'primary';
    window.currentVariant = variant;
    let assetUrl;
    if (variant === '3d_seam') {
      assetUrl = `/api/runs/${runId}/model_3d_seam.glb`;
    } else if (variant === '2_5d' || variant === 'terrain') {
      assetUrl = `/api/runs/${runId}/model_2_5d.glb`;
    } else if (variant === 'alt') {
      assetUrl = `/api/runs/${runId}/model_alt.glb`;
    } else if (variant === '3d') {
      assetUrl = `/api/runs/${runId}/model_3d.glb`;
    } else {
      assetUrl = `/api/runs/${runId}/model.glb`;
    }
    assetUrl += `?v=${Date.now()}`;

    const btnPrimary = document.getElementById('btn-variant-primary');
    const btnAlt = document.getElementById('btn-variant-alt');
    const btn3D = document.getElementById('btn-model-3d');
    const btn3DSeam = document.getElementById('btn-model-3d-seam');
    const btn25D = document.getElementById('btn-model-25d');

    const is25D = variant === 'primary' || variant === '2_5d' || variant === 'terrain';
    const is3D = variant === '3d';
    const is3DSeam = variant === '3d_seam';

    if (btnPrimary) btnPrimary.classList.toggle('active', variant === 'primary');
    if (btnAlt) btnAlt.classList.toggle('active', variant === 'alt');
    if (btn25D) btn25D.classList.toggle('active', is25D);
    if (btn3D) btn3D.classList.toggle('active', is3D);
    if (btn3DSeam) btn3DSeam.classList.toggle('active', is3DSeam);

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
              child.material.roughness = 0.8;
              child.material.metalness = 0.05;
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
              child.userData.originalMaterial = child.material;
              child.userData.currentShadedMaterial = child.material;
              child.userData.originalColor = child.material.color ? child.material.color.clone() : new THREE.Color(0xffffff);
              child.material.needsUpdate = true;
            }
            if (chkWireframe && chkWireframe.checked) {
              updateWireframeOverlay(child, true);
            }
            child.castShadow = false;
            child.receiveShadow = false;
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
  const chkAutorotate = document.getElementById('chk-autorotate');

  const btnShadeTextured = document.getElementById('btn-shade-textured');
  const btnShadeSemantic = document.getElementById('btn-shade-semantic');
  const btnShadeHeight = document.getElementById('btn-shade-height');
  const btnShadeWhite = document.getElementById('btn-shade-white');
  const btnCloseSemanticLeg = document.getElementById('btn-close-semantic-legend');
  const btnCloseHeightLeg = document.getElementById('btn-close-height-legend');

  if (btnShadeTextured) {
    btnShadeTextured.addEventListener('click', () => applyShadingMode('textured'));
  }
  if (btnShadeSemantic) {
    btnShadeSemantic.addEventListener('click', () => applyShadingMode('semantic'));
  }
  if (btnShadeHeight) {
    btnShadeHeight.addEventListener('click', () => applyShadingMode('height'));
  }
  if (btnShadeWhite) {
    btnShadeWhite.addEventListener('click', () => applyShadingMode('white'));
  }
  if (btnCloseSemanticLeg) {
    btnCloseSemanticLeg.addEventListener('click', () => {
      const leg = document.getElementById('semantic-legend');
      if (leg) leg.style.display = 'none';
    });
  }
  if (btnCloseHeightLeg) {
    btnCloseHeightLeg.addEventListener('click', () => {
      const leg = document.getElementById('height-legend');
      if (leg) leg.style.display = 'none';
    });
  }

  const btnVariantPrimary = document.getElementById('btn-variant-primary');
  const btnVariantAlt = document.getElementById('btn-variant-alt');
  const btnModel3D = document.getElementById('btn-model-3d');
  const btnModel3DSeam = document.getElementById('btn-model-3d-seam');
  const btnModel25D = document.getElementById('btn-model-25d');

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
  if (btnModel3D) {
    btnModel3D.addEventListener('click', () => {
      if (window.currentRunId) window.load3DModel(window.currentRunId, '3d');
    });
  }
  if (btnModel3DSeam) {
    btnModel3DSeam.addEventListener('click', () => {
      if (window.currentRunId) window.load3DModel(window.currentRunId, '3d_seam');
    });
  }
  if (btnModel25D) {
    btnModel25D.addEventListener('click', () => {
      if (window.currentRunId) window.load3DModel(window.currentRunId, '2_5d');
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
      const isVisible = e.target.checked;
      currentModel.traverse((child) => {
        if (child.isMesh) {
          updateWireframeOverlay(child, isVisible);
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
    if (!action) return;
    if (action.type === 'fit_view') {
      if (btnReset) btnReset.click();
      return;
    }
    if (!currentModel) return;
    const highlightColor = action.type === 'highlight_confidence' ? 0x3fb950
                         : action.type === 'toggle_masks' ? 0xf85149
                         : null;
    if (highlightColor != null) {
      currentModel.traverse((child) => {
        if (child.isMesh && child.material) {
          const mats = Array.isArray(child.material) ? child.material : [child.material];
          mats.forEach((mat) => {
            if (mat && mat.color) {
              const orig = mat.color.clone();
              mat.color.setHex(highlightColor);
              setTimeout(() => {
                if (mat && mat.color) mat.color.copy(orig);
              }, 2500);
            }
          });
        }
      });
    }
  };

  // Notify UI controller that viewer engine is ready
  window.viewerReady = true;
  window.dispatchEvent(new CustomEvent('viewer-ready'));
  const runToLoad = window.pendingRunId || window.currentRunId;
  if (runToLoad && !currentModel) {
    console.log('[dronemap] viewer initialized; loading run:', runToLoad);
    window.load3DModel(runToLoad);
    window.pendingRunId = null;
  }
} catch (err) {
  // Anything thrown above (WebGL context refused, shader compile failure, a
  // vendored Three.js file that 404'd) is surfaced rather than swallowed.
  reportEngineFailure(
    (err && err.message) ? err.message : 'Unknown error while initialising the 3D engine.'
  );
}
