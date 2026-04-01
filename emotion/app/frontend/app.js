/**
 * EMOSCAN — Facial Emotion Recognition Frontend
 *
 * Handles:
 * - Image upload with drag-and-drop
 * - Live camera feed with face detection overlay
 * - WebSocket streaming for real-time predictions
 * - Results rendering with animated confidence bars
 */

(function () {
  'use strict';

  // ── Emotion color map ──
  const EMOTION_COLORS = {
    angry:    '#ff3b3b',
    fear:     '#a855f7',
    happy:    '#facc15',
    neutral:  '#64748b',
    sad:      '#3b82f6',
    surprise: '#22d3ee',
    disgust:  '#22c55e',
  };

  // ── DOM refs ──
  const $ = (sel) => document.querySelector(sel);
  const statusDot    = $('.status-dot');
  const statusLabel  = $('#statusLabel');
  const btnUpload    = $('#btnUpload');
  const btnCamera    = $('#btnCamera');
  const modeSlider   = $('#modeSlider');
  const panelUpload  = $('#panelUpload');
  const panelCamera  = $('#panelCamera');
  const dropZone     = $('#dropZone');
  const fileInput    = $('#fileInput');
  const dropPreview  = $('#dropPreview');
  const previewImg   = $('#previewImg');
  const clearBtn     = $('#clearBtn');
  const analyzeBtn   = $('#analyzeBtn');
  const cameraFeed   = $('#cameraFeed');
  const overlayCanvas = $('#overlayCanvas');
  const flipBtn      = $('#flipBtn');
  const captureBtn   = $('#captureBtn');
  const streamToggle = $('#streamToggle');
  const resultsPanel = $('#resultsPanel');
  const resultsBody  = $('#resultsBody');
  const faceCount    = $('#faceCount');

  // ── State ──
  let currentMode = 'upload';
  let selectedFile = null;
  let cameraStream = null;
  let facingMode = 'user';
  let ws = null;
  let isStreaming = false;
  let streamInterval = null;

  // ── Status helpers ──
  function setStatus(label, state) {
    statusLabel.textContent = label;
    statusDot.className = 'status-dot' + (state ? ' ' + state : '');
  }

  // ── Mode switching ──
  function switchMode(mode) {
    currentMode = mode;

    btnUpload.classList.toggle('active', mode === 'upload');
    btnCamera.classList.toggle('active', mode === 'camera');
    modeSlider.classList.toggle('right', mode === 'camera');

    panelUpload.classList.toggle('active', mode === 'upload');
    panelCamera.classList.toggle('active', mode === 'camera');

    if (mode === 'camera') {
      startCamera();
    } else {
      stopCamera();
      stopStreaming();
    }
  }

  btnUpload.addEventListener('click', () => switchMode('upload'));
  btnCamera.addEventListener('click', () => switchMode('camera'));

  // ══════════════════════════════════════
  //  IMAGE UPLOAD
  // ══════════════════════════════════════

  dropZone.addEventListener('click', (e) => {
    if (e.target.closest('.clear-btn')) return;
    fileInput.click();
  });

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      handleFile(file);
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) {
      handleFile(fileInput.files[0]);
    }
  });

  function handleFile(file) {
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      dropPreview.classList.add('visible');
      analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  clearBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    selectedFile = null;
    previewImg.src = '';
    dropPreview.classList.remove('visible');
    analyzeBtn.disabled = true;
    fileInput.value = '';
    hideResults();
  });

  analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    analyzeBtn.classList.add('loading');
    analyzeBtn.disabled = true;
    setStatus('SCANNING', 'scanning');

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const res = await fetch('/predict/image', { method: 'POST', body: formData });

      if (!res.ok) throw new Error(`Server error: ${res.status}`);

      const data = await res.json();
      renderResults(data.predictions, data.faces_detected);
      setStatus('COMPLETE', '');
    } catch (err) {
      console.error('Analysis failed:', err);
      setStatus('ERROR', 'error');
      setTimeout(() => setStatus('READY', ''), 3000);
    } finally {
      analyzeBtn.classList.remove('loading');
      analyzeBtn.disabled = false;
    }
  });

  // ══════════════════════════════════════
  //  CAMERA
  // ══════════════════════════════════════

  async function startCamera() {
    try {
      if (cameraStream) {
        cameraStream.getTracks().forEach(t => t.stop());
      }

      cameraStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode, width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });

      cameraFeed.srcObject = cameraStream;
      setStatus('CAMERA ON', '');
    } catch (err) {
      console.error('Camera error:', err);
      setStatus('NO CAMERA', 'error');
    }
  }

  function stopCamera() {
    if (cameraStream) {
      cameraStream.getTracks().forEach(t => t.stop());
      cameraStream = null;
    }
  }

  flipBtn.addEventListener('click', () => {
    facingMode = facingMode === 'user' ? 'environment' : 'user';
    if (currentMode === 'camera') startCamera();
  });

  captureBtn.addEventListener('click', async () => {
    if (!cameraStream) return;

    setStatus('SCANNING', 'scanning');

    try {
      const canvas = document.createElement('canvas');
      canvas.width = cameraFeed.videoWidth;
      canvas.height = cameraFeed.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(cameraFeed, 0, 0);

      const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.85));
      const formData = new FormData();
      formData.append('file', blob, 'capture.jpg');

      const res = await fetch('/predict/image', { method: 'POST', body: formData });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);

      const data = await res.json();
      renderResults(data.predictions, data.faces_detected);
      drawOverlay(data.predictions);
      setStatus('COMPLETE', '');
    } catch (err) {
      console.error('Capture analysis failed:', err);
      setStatus('ERROR', 'error');
      setTimeout(() => setStatus('READY', ''), 3000);
    }
  });

  // ── WebSocket streaming ──
  streamToggle.addEventListener('click', () => {
    if (isStreaming) {
      stopStreaming();
    } else {
      startStreaming();
    }
  });

  function startStreaming() {
    if (!cameraStream) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/predict/stream`);

    ws.onopen = () => {
      isStreaming = true;
      streamToggle.classList.add('active');
      streamToggle.querySelector('.icon-stream-off').style.display = 'none';
      streamToggle.querySelector('.icon-stream-on').style.display = 'block';
      setStatus('STREAMING', 'scanning');

      streamInterval = setInterval(sendFrame, 200);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.error) {
        console.error('Stream error:', data.error);
        return;
      }
      renderResults(data.predictions, data.faces_detected);
      drawOverlay(data.predictions);
    };

    ws.onclose = () => {
      stopStreaming();
    };

    ws.onerror = (err) => {
      console.error('WebSocket error:', err);
      stopStreaming();
      setStatus('WS ERROR', 'error');
      setTimeout(() => setStatus('READY', ''), 3000);
    };
  }

  function stopStreaming() {
    isStreaming = false;
    if (streamInterval) {
      clearInterval(streamInterval);
      streamInterval = null;
    }
    if (ws) {
      ws.close();
      ws = null;
    }
    streamToggle.classList.remove('active');
    streamToggle.querySelector('.icon-stream-off').style.display = 'block';
    streamToggle.querySelector('.icon-stream-on').style.display = 'none';
    clearOverlay();
    setStatus('CAMERA ON', '');
  }

  function sendFrame() {
    if (!ws || ws.readyState !== WebSocket.OPEN || !cameraStream) return;

    const canvas = document.createElement('canvas');
    canvas.width = cameraFeed.videoWidth;
    canvas.height = cameraFeed.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(cameraFeed, 0, 0);

    const dataUrl = canvas.toDataURL('image/jpeg', 0.6);
    ws.send(dataUrl);
  }

  // ── Canvas overlay for face bounding boxes ──
  function drawOverlay(predictions) {
    const ctx = overlayCanvas.getContext('2d');
    overlayCanvas.width = cameraFeed.videoWidth;
    overlayCanvas.height = cameraFeed.videoHeight;
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    if (!predictions) return;

    predictions.forEach((pred) => {
      if (!pred.bbox) return;

      const [x, y, w, h] = pred.bbox;
      const emotions = pred.emotions;
      const dominant = getDominant(emotions);
      const color = EMOTION_COLORS[dominant] || '#22d3ee';

      // Draw bounding box
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);

      // Draw label background
      const label = `${dominant.toUpperCase()} ${Math.round(emotions[dominant] * 100)}%`;
      ctx.font = '500 13px "IBM Plex Mono", monospace';
      const metrics = ctx.measureText(label);
      const labelW = metrics.width + 12;
      const labelH = 22;
      const labelX = x;
      const labelY = y - labelH - 4;

      ctx.fillStyle = color + 'cc';
      ctx.beginPath();
      ctx.roundRect(labelX, labelY, labelW, labelH, 4);
      ctx.fill();

      ctx.fillStyle = '#000';
      ctx.fillText(label, labelX + 6, labelY + 15);

      // Corner accents
      const cornerLen = 10;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.setLineDash([]);

      // Top-left
      ctx.beginPath();
      ctx.moveTo(x, y + cornerLen); ctx.lineTo(x, y); ctx.lineTo(x + cornerLen, y);
      ctx.stroke();
      // Top-right
      ctx.beginPath();
      ctx.moveTo(x + w - cornerLen, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + cornerLen);
      ctx.stroke();
      // Bottom-left
      ctx.beginPath();
      ctx.moveTo(x, y + h - cornerLen); ctx.lineTo(x, y + h); ctx.lineTo(x + cornerLen, y + h);
      ctx.stroke();
      // Bottom-right
      ctx.beginPath();
      ctx.moveTo(x + w - cornerLen, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - cornerLen);
      ctx.stroke();
    });
  }

  function clearOverlay() {
    const ctx = overlayCanvas.getContext('2d');
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  }

  // ══════════════════════════════════════
  //  RESULTS RENDERING
  // ══════════════════════════════════════

  function getDominant(emotions) {
    let max = -1, dom = '';
    for (const [k, v] of Object.entries(emotions)) {
      if (v > max) { max = v; dom = k; }
    }
    return dom;
  }

  function renderResults(predictions, count) {
    resultsPanel.classList.add('visible');
    faceCount.textContent = count === 1 ? '1 FACE' : `${count} FACES`;

    resultsBody.innerHTML = '';

    predictions.forEach((pred, i) => {
      const emotions = pred.emotions;
      const dominant = getDominant(emotions);

      const card = document.createElement('div');
      card.className = 'face-card';
      card.style.animationDelay = `${i * 0.08}s`;

      const sorted = Object.entries(emotions)
        .sort((a, b) => b[1] - a[1]);

      card.innerHTML = `
        <div class="face-card-header">
          <span class="face-label">FACE ${i + 1}</span>
          <span class="dominant-emotion" data-emotion="${dominant}">
            ${dominant.toUpperCase()}
          </span>
        </div>
        <div class="emotion-bars">
          ${sorted.map(([name, val]) => {
            const pct = Math.round(val * 100);
            const isDom = name === dominant;
            return `
              <div class="emotion-row ${isDom ? 'dominant' : ''}" data-emotion="${name}">
                <span class="emotion-name">${name}</span>
                <div class="emotion-bar-track">
                  <div class="emotion-bar-fill" style="width: 0%"></div>
                </div>
                <span class="emotion-pct">${pct}%</span>
              </div>
            `;
          }).join('')}
        </div>
      `;

      resultsBody.appendChild(card);

      // Animate bars after a small delay
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          card.querySelectorAll('.emotion-bar-fill').forEach((bar, j) => {
            const pct = Math.round(sorted[j][1] * 100);
            bar.style.width = pct + '%';
          });
        });
      });
    });
  }

  function hideResults() {
    resultsPanel.classList.remove('visible');
    resultsBody.innerHTML = '';
  }

  // ── Init ──
  setStatus('READY', '');
})();
