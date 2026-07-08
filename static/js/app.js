/*
  app.js
  ------
  STEP-BY-STEP EXPLANATION:

  1. TAB SWITCHING: clicking "Live Camera" / "Upload Photo" / "About"
     just shows/hides the matching <section>. Pure vanilla JS, no
     framework needed -- keeps this beginner-friendly and dependency-free.

  2. LIVE CAMERA:
     - navigator.mediaDevices.getUserMedia() asks the browser for
       webcam access (built into every modern browser -- Chrome,
       Firefox, Safari, Edge, on desktop AND mobile).
     - Once the camera is running, we grab a frame every 2 seconds,
       draw it onto a hidden <canvas>, convert that canvas to a
       base64 JPEG string, and POST it to our Flask "/predict" route.
     - The server sends back {label, confidence} and we update the
       badge on screen.

  3. UPLOAD PHOTO:
     - Works the same way, but instead of a live video frame, we read
       whatever image file the user picked or dragged in.

  This file talks to app.py ONLY through the "/predict" endpoint --
  everything else is pure browser code, so it works identically on
  any operating system.
*/

// ---------- TAB SWITCHING ----------
const tabButtons = document.querySelectorAll(".tab-btn");
const panels = {
  live: document.getElementById("panel-live"),
  upload: document.getElementById("panel-upload"),
  about: document.getElementById("panel-about"),
};

tabButtons.forEach(btn => {
  btn.addEventListener("click", () => {
    tabButtons.forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    Object.values(panels).forEach(p => p.classList.remove("active"));
    panels[btn.dataset.tab].classList.add("active");

    if (btn.dataset.tab === "live") startCameraIfNeeded();
    if (btn.dataset.tab === "about") loadAccuracyIfAvailable();
  });
});

// ---------- SHARED: send a base64 image to the backend ----------
async function sendToBackend(base64Image) {
  const response = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: base64Image }),
  });
  return response.json();
}

function paintBadge(badgeEl, result) {
  if (result.error) {
    badgeEl.className = "result-badge";
    badgeEl.innerHTML = `<span class="result-label">${result.error}</span>`;
    return;
  }
  const cls = result.label === "Depressed" ? "depressed" : "not-depressed";
  badgeEl.className = `result-badge ${cls}`;
  badgeEl.innerHTML = `<span class="result-label">${result.label} — ${result.confidence}% confidence</span>`;
}

// ---------- LIVE CAMERA ----------
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const startBtn = document.getElementById("startBtn");
const liveResult = document.getElementById("liveResult");

let cameraStarted = false;
let liveIntervalId = null;

async function startCameraIfNeeded() {
  if (cameraStarted) return;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    cameraStarted = true;
    startBtn.textContent = "Camera running";
    startBtn.disabled = true;

    liveIntervalId = setInterval(captureAndAnalyzeLiveFrame, 1500);
  } catch (err) {
    liveResult.innerHTML = `<span class="result-label">Camera access denied or unavailable.</span>`;
  }
}

// Keeps the last few predictions so we can smooth out flickering
let predictionHistory = [];
const HISTORY_SIZE = 3; // how many recent predictions to consider

function captureAndAnalyzeLiveFrame() {
  if (!video.videoWidth) return; // camera not ready yet

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const base64Image = canvas.toDataURL("image/jpeg", 0.85);
  sendToBackend(base64Image).then(result => {
    if (result.error) {
      paintBadge(liveResult, result);
      return;
    }

    // Add this prediction to our short history
    predictionHistory.push(result.label);
    if (predictionHistory.length > HISTORY_SIZE) {
      predictionHistory.shift(); // remove oldest
    }

    // Count how many times each label appears in recent history
    const depressedCount = predictionHistory.filter(l => l === "Depressed").length;
    const notDepressedCount = predictionHistory.length - depressedCount;

    // Only show a label once it has a clear majority in recent history
    const smoothedLabel = depressedCount > notDepressedCount ? "Depressed" : "Not Depressed";

    paintBadge(liveResult, {
      label: smoothedLabel,
      confidence: result.confidence
    });
  });
}
startBtn.addEventListener("click", startCameraIfNeeded);

// ---------- UPLOAD PHOTO ----------
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const dropLabel = document.getElementById("dropLabel");
const previewImg = document.getElementById("previewImg");
const analyzeBtn = document.getElementById("analyzeBtn");
const uploadResult = document.getElementById("uploadResult");

let selectedBase64 = null;

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    selectedBase64 = e.target.result;
    previewImg.src = selectedBase64;
    previewImg.style.display = "block";
    dropLabel.textContent = file.name;
    analyzeBtn.disabled = false;
  };
  reader.readAsDataURL(file);
});

// allow drag-and-drop too
["dragover", "drop"].forEach(evt =>
  dropZone.addEventListener(evt, e => e.preventDefault())
);
dropZone.addEventListener("drop", e => {
  const file = e.dataTransfer.files[0];
  if (file) {
    fileInput.files = e.dataTransfer.files;
    fileInput.dispatchEvent(new Event("change"));
  }
});

analyzeBtn.addEventListener("click", () => {
  if (!selectedBase64) return;
  uploadResult.innerHTML = `<span class="result-label">Analyzing…</span>`;
  sendToBackend(selectedBase64).then(result => paintBadge(uploadResult, result));
});

// ---------- ABOUT TAB: show real test accuracy if backend provides it ----------
function loadAccuracyIfAvailable() {
  const el = document.getElementById("accuracyStat");
  fetch("/accuracy")
    .then(r => r.json())
    .then(data => {
      if (data && data.accuracy) el.textContent = data.accuracy + "%";
    })
    .catch(() => {}); // silently ignore if the route doesn't exist yet
}

// Start the camera automatically on first load since "Live Camera" is the default tab
startCameraIfNeeded();
