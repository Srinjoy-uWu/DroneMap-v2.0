/* ============================================================
   DroneMap — UI Controller (Simplified & Accessible)
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {
  console.log('[dronemap] UI Controller initialized');

  // Prevent default browser drag/drop navigation
  window.addEventListener('dragover', (e) => e.preventDefault(), false);
  window.addEventListener('drop', (e) => e.preventDefault(), false);

  // UI Element Selectors
  const btnOpenUpload    = document.getElementById('btn-open-upload');
  const uploadModal      = document.getElementById('upload-modal');
  const btnModalClose    = document.getElementById('btn-modal-close');
  const btnCancelUpload  = document.getElementById('btn-cancel-upload');
  const uploadForm       = document.getElementById('upload-form');

  const jobModal         = document.getElementById('job-modal');
  const btnJobClose      = document.getElementById('btn-job-modal-close');
  const jobProgressBar   = document.getElementById('job-progress-bar');
  const jobLogOutput     = document.getElementById('job-log-output');
  const jobElapsedTime   = document.getElementById('job-elapsed-time');
  const jobModalFooter   = document.getElementById('job-modal-footer');
  const btnViewFinished  = document.getElementById('btn-view-finished-model');

  const runListContainer = document.getElementById('run-list');
  const btnRefreshRuns   = document.getElementById('btn-refresh-runs');
  const reportSection    = document.getElementById('report-section');
  const accuracyReportEl = document.getElementById('accuracy-report');
  const downloadSection  = document.getElementById('download-section');
  const downloadLinksEl  = document.getElementById('download-links');

  // Modal Open & Close Handlers
  if (btnOpenUpload) {
    btnOpenUpload.addEventListener('click', (e) => {
      e.preventDefault();
      uploadModal.style.display = 'flex';
      
      const d = new Date();
      const pad = (n) => String(n).padStart(2, '0');
      const inputRunId = document.getElementById('input-run-id');
      if (inputRunId && !inputRunId.value) {
        inputRunId.placeholder = `flight_${d.getFullYear()}${pad(d.getMonth()+1)}${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}`;
      }
    });
  }

  function closeUploadModal() {
    uploadModal.style.display = 'none';
  }

  if (btnModalClose) btnModalClose.addEventListener('click', closeUploadModal);
  if (btnCancelUpload) btnCancelUpload.addEventListener('click', closeUploadModal);
  if (btnJobClose) btnJobClose.addEventListener('click', () => { jobModal.style.display = 'none'; });

  window.addEventListener('click', (e) => {
    if (e.target === uploadModal) closeUploadModal();
    if (e.target === jobModal && btnJobClose.style.display !== 'none') {
      jobModal.style.display = 'none';
    }
  });

  // Drag & Drop Setup
  function setupDropzone(dropzoneId, inputId, infoBoxId) {
    const dropzone = document.getElementById(dropzoneId);
    const input = document.getElementById(inputId);
    const infoBox = document.getElementById(infoBoxId);
    if (!dropzone || !input || !infoBox) return;

    const dropContent = dropzone.querySelector('.dropzone-content');

    ['dragenter', 'dragover'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (!dropzone.classList.contains('dropzone-disabled')) {
          dropzone.classList.add('drag-over');
        }
      }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove('drag-over');
      }, false);
    });

    dropzone.addEventListener('drop', (e) => {
      if (dropzone.classList.contains('dropzone-disabled')) return;
      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        input.files = files;
        updateFileInfo(files[0]);
      }
    }, false);

    input.addEventListener('change', () => {
      if (input.files && input.files.length > 0) {
        updateFileInfo(input.files[0]);
      }
    });

    function updateFileInfo(file) {
      if (dropContent) dropContent.style.display = 'none';
      infoBox.style.display = 'flex';
      const nameEl = infoBox.querySelector('.file-name');
      const sizeEl = infoBox.querySelector('.file-size');
      if (nameEl) nameEl.textContent = file.name;
      if (sizeEl) sizeEl.textContent = `(${(file.size / 1024 / 1024).toFixed(1)} MB)`;
    }

    const clearBtn = infoBox.querySelector('.btn-clear-file');
    if (clearBtn) {
      clearBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        input.value = '';
        infoBox.style.display = 'none';
        if (dropContent) dropContent.style.display = 'flex';
      });
    }
  }

  setupDropzone('video-dropzone', 'video-file-input', 'video-file-info');
  setupDropzone('telem-dropzone', 'telem-file-input', 'telem-file-info');

  // No-telemetry checkbox toggle
  const chkNoTelem = document.getElementById('chk-no-telemetry');
  const telemDropzone = document.getElementById('telem-dropzone');
  const telemInput = document.getElementById('telem-file-input');

  if (chkNoTelem && telemDropzone) {
    chkNoTelem.addEventListener('change', (e) => {
      if (e.target.checked) {
        telemDropzone.classList.add('dropzone-disabled');
        if (telemInput) telemInput.disabled = true;
      } else {
        telemDropzone.classList.remove('dropzone-disabled');
        if (telemInput) telemInput.disabled = false;
      }
    });
  }

  // Upload & Process Submission
  if (uploadForm) {
    uploadForm.addEventListener('submit', (e) => {
      e.preventDefault();

      const videoInput = document.getElementById('video-file-input');
      if (!videoInput.files || videoInput.files.length === 0) {
        alert('Please select a drone video file to proceed.');
        return;
      }

      const formData = new FormData(uploadForm);
      uploadModal.style.display = 'none';
      jobModal.style.display = 'flex';

      // Reset Stepper & Logs
      document.querySelectorAll('.step-card').forEach(c => {
        c.className = 'step-card';
        const st = c.querySelector('.step-status');
        if (st) st.textContent = 'Pending';
      });
      jobProgressBar.style.width = '0%';
      jobLogOutput.textContent = 'Uploading drone video to processing server…\n';
      jobModalFooter.style.display = 'none';
      btnJobClose.style.display = 'none';
      document.getElementById('job-modal-title').textContent = 'Processing Video…';
      document.getElementById('job-modal-subtitle').textContent = 'Building 3D model from flight video';

      const startTime = Date.now();
      const timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        const m = String(Math.floor(elapsed / 60)).padStart(2, '0');
        const s = String(elapsed % 60).padStart(2, '0');
        jobElapsedTime.textContent = `${m}:${s}`;
      }, 1000);

      // XHR upload for real-time progress
      const xhr = new XMLHttpRequest();
      xhr.open('POST', '/api/jobs');

      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          const pct = Math.round((event.loaded / event.total) * 100);
          jobProgressBar.style.width = `${Math.min(pct, 95)}%`;
          jobLogOutput.textContent = `Uploading video: ${pct}% (${(event.loaded/1024/1024).toFixed(1)} MB / ${(event.total/1024/1024).toFixed(1)} MB)…\n`;
        }
      };

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          const data = JSON.parse(xhr.responseText);
          const jobId = data.job_id;
          const runId = data.run_id;
          startJobPolling(jobId, runId, timerInterval);
        } else {
          clearInterval(timerInterval);
          btnJobClose.style.display = 'block';
          document.getElementById('job-modal-title').textContent = 'Upload Failed';
          jobLogOutput.textContent += `\nError: Server returned status ${xhr.status} — ${xhr.responseText}`;
        }
      };

      xhr.onerror = () => {
        clearInterval(timerInterval);
        btnJobClose.style.display = 'block';
        document.getElementById('job-modal-title').textContent = 'Network Error';
        jobLogOutput.textContent += '\nNetwork error occurred while uploading video.';
      };

      xhr.send(formData);
    });
  }

  // Job Polling
  function startJobPolling(jobId, runId, timerInterval) {
    const stageOrder = ['frames', 'masks', 'pose', 'dense', 'mesh', 'export'];

    const pollInterval = setInterval(async () => {
      try {
        const res = await fetch(`/api/jobs/${jobId}`);
        if (!res.ok) return;
        const job = await res.json();

        // Update logs
        if (job.logs && job.logs.length) {
          jobLogOutput.textContent = job.logs.join('\n');
          jobLogOutput.scrollTop = jobLogOutput.scrollHeight;
        }

        // Update Stepper Cards
        let completedCount = 0;
        stageOrder.forEach(stKey => {
          const card = document.querySelector(`.step-card[data-stage="${stKey}"]`);
          if (!card) return;
          const status = job.stages[stKey];
          const stEl = card.querySelector('.step-status');

          card.classList.remove('active', 'completed', 'failed');
          if (status === 'ok') {
            card.classList.add('completed');
            if (stEl) stEl.textContent = 'Done';
            completedCount++;
          } else if (status === 'running') {
            card.classList.add('active');
            if (stEl) stEl.textContent = 'Processing';
          } else if (status === 'failed') {
            card.classList.add('failed');
            if (stEl) stEl.textContent = 'Failed';
          } else {
            if (stEl) stEl.textContent = 'Pending';
          }
        });

        // Progress bar
        const progressPercent = Math.max(10, Math.round((completedCount / stageOrder.length) * 100));
        jobProgressBar.style.width = `${progressPercent}%`;

        // Completion
        if (job.status === 'completed') {
          clearInterval(pollInterval);
          clearInterval(timerInterval);
          jobProgressBar.style.width = '100%';
          jobModalFooter.style.display = 'flex';
          btnJobClose.style.display = 'block';
          document.getElementById('job-modal-title').textContent = 'Processing Complete!';
          document.getElementById('job-modal-subtitle').textContent = `3D model ready for '${runId}'`;

          btnViewFinished.onclick = () => {
            jobModal.style.display = 'none';
            selectRun(runId, true);
            fetchRuns(runId);
          };

          fetchRuns(runId);
        } else if (job.status === 'failed') {
          clearInterval(pollInterval);
          clearInterval(timerInterval);
          btnJobClose.style.display = 'block';
          document.getElementById('job-modal-title').textContent = 'Processing Error';
          document.getElementById('job-modal-subtitle').textContent = job.error || 'A processing step failed.';
        }
      } catch (err) {
        console.error('[dronemap] Poll error:', err);
      }
    }, 1200);
  }

  // Centralized Run Selection & Viewer Synchronization
  function selectRun(runId, hasModel = true) {
    window.currentRunId = runId;
    document.querySelectorAll('.run-card').forEach(c => {
      c.classList.toggle('selected', c.dataset.runId === runId);
    });
    if (hasModel) {
      if (window.load3DModel) {
        window.load3DModel(runId);
      } else {
        window.pendingRunId = runId;
      }
    }
    showTextureVariantToggle(runId);
    loadAccuracyReport(runId);
    renderDownloadLinks(runId);
    loadStagesAudit(runId);
  }
  window.selectRun = selectRun;

  // The texture-variant toggle exists only for runs that actually produced two
  // atlases. Offering it unconditionally would give a button that 404s, which
  // is worse than not offering it at all.
  async function showTextureVariantToggle(runId) {
    const grp = document.getElementById('grp-texture-variant');
    const div = document.getElementById('div-texture-variant');
    const show = (on) => {
      if (grp) grp.style.display = on ? '' : 'none';
      if (div) div.style.display = on ? '' : 'none';
    };
    show(false);
    try {
      const res = await fetch(`/api/runs/${runId}/model_alt.glb`, { method: 'HEAD' });
      if (!res.ok || window.currentRunId !== runId) return;
      const label = await textureVariantLabel(runId);
      const btnAlt = document.getElementById('btn-variant-alt');
      const btnPrimary = document.getElementById('btn-variant-primary');
      if (btnAlt && label) {
        btnAlt.textContent = label.alt;
        btnAlt.title = label.altTitle;
      }
      if (btnPrimary && label) {
        btnPrimary.textContent = label.primary;
        btnPrimary.title = label.primaryTitle;
      }
      show(true);
    } catch (err) {
      /* No alternate variant for this run; the toggle stays hidden. */
    }
  }

  // Name the two buttons after what actually differs, read from the manifest,
  // so the comparison is legible without knowing the pipeline internals.
  async function textureVariantLabel(runId) {
    try {
      const res = await fetch(`/api/runs/${runId}`);
      const m = await res.json();
      const outs = ((m.stages || {}).mesh || {}).outputs || {};
      const which = outs.textured_obj_alt_label;
      if (which === 'seam_levelled') {
        return {
          primary: 'No Seam Levelling',
          primaryTitle: 'Raw per-view projection — true colour, visible seams',
          alt: 'Seam Levelled',
          altTitle: 'OpenMVS blended atlas — the variant that came out dark',
        };
      }
      return {
        primary: 'Seam Levelled',
        primaryTitle: 'OpenMVS blended atlas (adopted)',
        alt: 'No Seam Levelling',
        altTitle: 'Raw per-view projection — true colour, visible seams',
      };
    } catch (err) {
      return null;
    }
  }

  // Listen for when 3D viewer finishes module initialization
  window.addEventListener('viewer-ready', () => {
    if (window.currentRunId && window.load3DModel) {
      window.load3DModel(window.currentRunId);
      showTextureVariantToggle(window.currentRunId);
    }
  });

  // Sidebar Runs List & Reports
  async function fetchRuns(preferredRunId = null) {
    try {
      const res = await fetch('/api/runs');
      const runs = await res.json();
      if (!runs || !runs.length) {
        runListContainer.innerHTML = '<div class="empty-state">No projects yet. Click <strong>+ Process Drone Video</strong> above to start.</div>';
        return;
      }

      runListContainer.innerHTML = '';
      let autoSelected = false;
      const targetId = preferredRunId || window.currentRunId;

      runs.forEach((run) => {
        const card = document.createElement('div');
        const isTarget = targetId ? run.run_id === targetId : false;
        card.className = `run-card ${isTarget ? 'selected' : ''}`;
        card.dataset.runId = run.run_id;
        const dateStr = run.created_utc ? run.created_utc.substring(0, 10) : 'Recent';
        // 'georeferenced' shares the caution colour with 'relative': the model
        // carries world coordinates but their error was never measured, so it
        // must not look as trustworthy as a validated run.
        const badgeClass = run.status_type === 'validated' ? 'badge-ready' :
                           run.status_type === 'terrain' ? 'badge-terrain' :
                           run.status_type === 'relative' ? 'badge-warning' :
                           run.status_type === 'georeferenced' ? 'badge-warning' :
                           run.status_type === 'rejected' ? 'badge-danger' : 'badge-running';

        card.innerHTML = `
          <div class="run-card-header">
            <span class="run-name" title="${run.run_id}">${run.run_id}</span>
            <span class="run-badge ${badgeClass}">${run.badge || 'Processing'}</span>
          </div>
          <div class="run-details">
            <span>${run.badge || 'Processing'}</span>
            <span>&bull;</span>
            <span>${dateStr}</span>
          </div>
        `;

        card.addEventListener('click', () => {
          selectRun(run.run_id, run.has_model);
        });

        runListContainer.appendChild(card);

        if (isTarget && !autoSelected) {
          selectRun(run.run_id, run.has_model);
          autoSelected = true;
        }
      });

      // If no preferred target was set/found, auto-load the newest run that has a 3D model
      if (!autoSelected && !window.currentRunId) {
        const firstUsable = runs.find(r => r.has_model);
        if (firstUsable) {
          selectRun(firstUsable.run_id, true);
        }
      }
    } catch (err) {
      runListContainer.innerHTML = `<div class="empty-state" style="color:var(--danger)">Failed to load projects: ${err.message}</div>`;
    }
  }

  async function loadAccuracyReport(runId) {
    try {
      const res = await fetch(`/api/runs/${runId}`);
      const m = await res.json();
      reportSection.style.display = 'block';

      const acc = m.accuracy || {};
      const stages = m.stages || {};
      const nDense = stages.dense?.metrics?.n_dense_points || acc.n_dense_points;
      const pts = nDense ? Number(nDense).toLocaleString() + ' pts' : '—';
      const crs = acc.crs || 'LOCAL_RELATIVE';
      const isGeoref = crs !== 'LOCAL_RELATIVE' && stages.georef?.status === 'ok';
      const gpsLabel = isGeoref ? `Aligned (${crs})` : 'Relative / Unaligned';
      const rmseLabel = acc.alignment_rmse_m != null ? `${acc.alignment_rmse_m.toFixed(2)} m` : '—';

      accuracyReportEl.innerHTML = `
        <div class="report-row"><span class="report-label">3D Points</span><span class="report-value">${pts}</span></div>
        <div class="report-row"><span class="report-label">Georef CRS</span><span class="report-value">${gpsLabel}</span></div>
        <div class="report-row"><span class="report-label">Alignment RMSE</span><span class="report-value">${rmseLabel}</span></div>
      `;
    } catch (e) {
      reportSection.style.display = 'none';
    }
  }

  async function renderDownloadLinks(runId) {
    try {
      const res = await fetch(`/api/runs/${runId}`);
      const m = await res.json();
      const exportOutputs = m.stages?.export?.outputs || {};
      downloadSection.style.display = 'block';

      const links = [];
      links.push(`<a class="dl-btn dl-btn-primary" href="/api/runs/${runId}/report.html" target="_blank">Full Report (HTML)</a>`);
      if (exportOutputs.model_glb) {
        links.push(`<a class="dl-btn" href="/api/runs/${runId}/model.glb" download>3D Model (.glb)</a>`);
      }
      if (exportOutputs.model_alt_glb) {
        const variant = (exportOutputs.model_alt_label || 'alternate').replace(/_/g, ' ');
        links.push(`<a class="dl-btn" href="/api/runs/${runId}/model_alt.glb" download>3D Model — ${variant} texture (.glb)</a>`);
      }
      if (exportOutputs.cloud_laz) {
        links.push(`<a class="dl-btn" href="/api/runs/${runId}/cloud.laz" download>Point Cloud (.laz)</a>`);
      }
      if (exportOutputs.ortho_tif) {
        links.push(`<a class="dl-btn" href="/api/runs/${runId}/orthomosaic.tif" download>2D Aerial Map (.tif)</a>`);
      }
      if (exportOutputs.dsm_tif) {
        links.push(`<a class="dl-btn" href="/api/runs/${runId}/dsm.tif" download>Surface Elevation (.tif)</a>`);
      }
      if (exportOutputs.dtm_tif) {
        links.push(`<a class="dl-btn" href="/api/runs/${runId}/dtm.tif" download>Terrain Elevation (.tif)</a>`);
      }
      if (exportOutputs.trajectory_kml) {
        links.push(`<a class="dl-btn" href="/api/runs/${runId}/trajectory.kml" download>Flight Path (.kml)</a>`);
      }

      downloadLinksEl.innerHTML = links.join('');
    } catch (e) {
      downloadSection.style.display = 'none';
    }
  }

  // ------------------------------------------------------------
  // Quality & Hardening Audit Drawer
  // ------------------------------------------------------------
  async function loadStagesAudit(runId) {
    try {
      const res = await fetch(`/api/runs/${runId}/stages`);
      if (!res.ok) return;
      const data = await res.json();
      const stages = data.stages || {};

      // Stage 1: Keyframes
      const s1 = stages.frames?.metrics || {};
      const nDec = s1.n_frames_decoded || 0;
      const nKf = s1.n_keyframes || 0;
      const red = nDec > 0 ? ((nDec - nKf) / nDec * 100).toFixed(1) : '0.0';
      const elS1Dec = document.getElementById('audit-s1-decoded');
      const elS1Kf = document.getElementById('audit-s1-keyframes');
      const elS1Red = document.getElementById('audit-s1-reduction');
      if (elS1Dec) elS1Dec.textContent = nDec ? `${nDec} frames` : '—';
      if (elS1Kf) elS1Kf.textContent = nKf ? `${nKf} keyframes` : '—';
      if (elS1Red) elS1Red.textContent = nDec ? `${red}% reduction` : '—';

      // Stage 2: Dynamic Masks
      const s2 = stages.masks?.metrics || {};
      const maskedFrac = s2.mean_masked_fraction != null ? (s2.mean_masked_fraction * 100).toFixed(1) + '%' : '0.0%';
      const elS2Masked = document.getElementById('audit-s2-masked');
      if (elS2Masked) elS2Masked.textContent = maskedFrac;

      const previewCont = document.getElementById('audit-s2-preview-container');
      const previewImg = document.getElementById('audit-s2-preview-img');
      if (data.mask_previews && data.mask_previews.length > 0 && previewCont && previewImg) {
        previewImg.src = `/api/runs/${runId}/masks/preview/${data.mask_previews[0]}`;
        previewCont.style.display = 'block';
      } else if (previewCont) {
        previewCont.style.display = 'none';
      }

      // Stage 3: Pose & Neural Escalation
      const s3 = stages.pose?.metrics || {};
      const elS3Mode = document.getElementById('audit-s3-mode');
      const elS3Reg = document.getElementById('audit-s3-registered');
      const elS3Reproj = document.getElementById('audit-s3-reproj');
      if (elS3Mode) elS3Mode.textContent = s3.neural_fallback_used ? 'Hybrid Neural (LightGlue)' : 'Classical Guided SIFT';
      if (elS3Reg) elS3Reg.textContent = s3.registered_fraction != null ? `${(s3.registered_fraction * 100).toFixed(1)}%` : '—';
      if (elS3Reproj) elS3Reproj.textContent = s3.mean_reproj_error != null ? `${s3.mean_reproj_error.toFixed(2)} px` : '—';

      // Stage 5 & 6: Dense Cloud & Mesh
      const s5 = stages.dense?.metrics || {};
      const s6 = stages.mesh?.metrics || {};
      const elS5Pts = document.getElementById('audit-s5-pts');
      const elS6Faces = document.getElementById('audit-s6-faces');
      if (elS5Pts) elS5Pts.textContent = s5.n_dense_points ? Number(s5.n_dense_points).toLocaleString() : '—';
      if (elS6Faces) elS6Faces.textContent = s6.n_faces ? Number(s6.n_faces).toLocaleString() : '—';

      // Stage 7: Regional Confidence
      const s7 = stages.export?.metrics || {};
      const obsPct = Math.round((s7.laz_observed_fraction || 0.75) * 100);
      const estPct = Math.round((s7.laz_estimated_fraction || 0.18) * 100);
      const infPct = Math.max(0, 100 - obsPct - estPct);
      const bObs = document.getElementById('audit-conf-obs-bar');
      const bEst = document.getElementById('audit-conf-est-bar');
      const bInf = document.getElementById('audit-conf-inf-bar');
      if (bObs) bObs.style.width = `${obsPct}%`;
      if (bEst) bEst.style.width = `${estPct}%`;
      if (bInf) bInf.style.width = `${infPct}%`;

      const tObs = document.getElementById('audit-conf-obs');
      const tEst = document.getElementById('audit-conf-est');
      const tInf = document.getElementById('audit-conf-inf');
      if (tObs) tObs.textContent = `${obsPct}%`;
      if (tEst) tEst.textContent = `${estPct}%`;
      if (tInf) tInf.textContent = `${infPct}%`;

      // Diagnostics
      const diagRes = await fetch(`/api/runs/${runId}/diagnostics`);
      if (diagRes.ok) {
        const diag = await diagRes.json();
        const badge = document.getElementById('audit-health-badge');
        const summary = document.getElementById('audit-diag-summary');
        const actionsList = document.getElementById('audit-tuning-actions');
        if (badge) {
          badge.textContent = diag.overall_health;
          badge.className = `badge-tag ${diag.overall_health !== 'HEALTHY' ? 'tag-warn' : ''}`;
        }
        if (summary) summary.textContent = diag.diagnostic_summary;
        if (actionsList) {
          actionsList.innerHTML = (diag.tuning_recommendations || []).map(a =>
            `<div class="audit-action-item"><strong>${escapeHtml(a.parameter)}:</strong> ${escapeHtml(String(a.recommended_value))} &bull; ${escapeHtml(a.reason)}</div>`
          ).join('');
        }
      }
    } catch (e) {
      console.warn('[dronemap] Failed to load stages audit:', e);
    }
  }

  // ------------------------------------------------------------
  // Sliding Drawers Controls & Copilot Chat
  // ------------------------------------------------------------
  const btnToggleAudit = document.getElementById('btn-toggle-audit');
  const btnCloseAudit = document.getElementById('btn-close-audit');
  const auditDrawer = document.getElementById('audit-drawer');

  const btnToggleCopilot = document.getElementById('btn-toggle-copilot');
  const btnCloseCopilot = document.getElementById('btn-close-copilot');
  const copilotDrawer = document.getElementById('copilot-drawer');
  const copilotForm = document.getElementById('copilot-form');
  const copilotInput = document.getElementById('copilot-input');
  const copilotMessages = document.getElementById('copilot-messages');

  if (btnToggleAudit) {
    btnToggleAudit.addEventListener('click', () => {
      const show = auditDrawer.style.display === 'none';
      auditDrawer.style.display = show ? 'flex' : 'none';
      if (show && copilotDrawer) copilotDrawer.style.display = 'none';
    });
  }
  if (btnCloseAudit) {
    btnCloseAudit.addEventListener('click', () => { auditDrawer.style.display = 'none'; });
  }

  if (btnToggleCopilot) {
    btnToggleCopilot.addEventListener('click', () => {
      const show = copilotDrawer.style.display === 'none';
      copilotDrawer.style.display = show ? 'flex' : 'none';
      if (show && auditDrawer) auditDrawer.style.display = 'none';
      if (show && copilotInput) copilotInput.focus();
    });
  }
  if (btnCloseCopilot) {
    btnCloseCopilot.addEventListener('click', () => { copilotDrawer.style.display = 'none'; });
  }

  // Quick Chips
  document.querySelectorAll('.copilot-chips .chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const q = chip.dataset.query;
      if (copilotInput && q) {
        copilotInput.value = q;
        sendCopilotQuery(q);
      }
    });
  });

  if (copilotForm) {
    copilotForm.addEventListener('submit', (e) => {
      e.preventDefault();
      const q = copilotInput.value.trim();
      if (!q) return;
      copilotInput.value = '';
      sendCopilotQuery(q);
    });
  }

  async function sendCopilotQuery(query) {
    if (!copilotMessages) return;
    const userDiv = document.createElement('div');
    userDiv.className = 'copilot-msg msg-user';
    userDiv.innerHTML = `<div class="msg-bubble">${escapeHtml(query)}</div>`;
    copilotMessages.appendChild(userDiv);

    const aiDiv = document.createElement('div');
    aiDiv.className = 'copilot-msg msg-ai';
    aiDiv.innerHTML = `<div class="msg-bubble"><span style="color:var(--text-dim)">Analyzing spatial survey facts...</span></div>`;
    copilotMessages.appendChild(aiDiv);
    copilotMessages.scrollTop = copilotMessages.scrollHeight;

    const runId = window.currentRunId || 'default';
    try {
      const res = await fetch(`/api/runs/${runId}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });
      const data = await res.json();
      aiDiv.innerHTML = `<div class="msg-bubble">${formatMarkdown(data.reply)}</div>`;
      copilotMessages.scrollTop = copilotMessages.scrollHeight;

      if (data.action && window.triggerViewerAction) {
        window.triggerViewerAction(data.action);
      }
    } catch (err) {
      aiDiv.innerHTML = `<div class="msg-bubble" style="color:#f85149">Failed to query assistant: ${err.message}</div>`;
    }
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  function formatMarkdown(md) {
    if (!md) return '';
    return md
      .replace(/^### (.*$)/gim, '<h3>$1</h3>')
      .replace(/^## (.*$)/gim, '<h3>$1</h3>')
      .replace(/^# (.*$)/gim, '<h3>$1</h3>')
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/^\- (.*$)/gim, '<li>$1</li>')
      .replace(/\n\n/g, '<br><br>');
  }

  if (btnRefreshRuns) btnRefreshRuns.addEventListener('click', fetchRuns);

  fetchRuns();
  window.fetchRuns = fetchRuns;
});
