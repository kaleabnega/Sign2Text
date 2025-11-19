const videoElement = document.getElementById("input-video");
const canvasElement = document.getElementById("output-canvas");
const canvasCtx = canvasElement.getContext("2d");
const statusOverlay = document.getElementById("status-overlay");
const predictionEl = document.getElementById("prediction");
const confidenceEl = document.getElementById("confidence");
const logOutput = document.getElementById("log-output");
const recordButton = document.getElementById("record-button");
const predictButton = document.getElementById("predict-button");
const resetButton = document.getElementById("reset-button");
const apiUrlInput = document.getElementById("api-url");
const progressFill = document.getElementById("progress-fill");
const dialog = document.getElementById("prediction-dialog");
const dialogPrediction = document.getElementById("dialog-prediction");
const dialogConfidence = document.getElementById("dialog-confidence");
const dialogClose = document.getElementById("dialog-close");

const SEQUENCE_LENGTH = 32;
const FEATURE_DIM = 2 * 21 * 3;
const rawFrames = [];
const buffer = [];
let isPredicting = false;
let isRecording = false;
let recordAnimationId = null;
const RECORD_DURATION_MS = 5000;

function log(message) {
  const timestamp = new Date().toLocaleTimeString();
  logOutput.textContent = `[${timestamp}] ${message}\n${logOutput.textContent}`;
}

function normalizeLandmarks(coords) {
  const mean = coords
    .reduce((acc, [x, y, z]) => [acc[0] + x, acc[1] + y, acc[2] + z], [0, 0, 0])
    .map((value) => value / coords.length);
  const centered = coords.map(([x, y, z]) => [x - mean[0], y - mean[1], z - mean[2]]);
  const norm =
    Math.sqrt(centered.reduce((sum, [x, y, z]) => sum + x * x + y * y + z * z, 0)) + 1e-6;
  return centered.map(([x, y, z]) => [x / norm, y / norm, z / norm]);
}

function zeroPadHand() {
  return Array.from({ length: 21 }, () => [0, 0, 0]);
}

function flattenHands(left, right) {
  const flat = [];
  [...left, ...right].forEach(([x, y, z]) => flat.push(x, y, z));
  return flat;
}

function updateProgress(ratio) {
  progressFill.style.width = `${Math.min(ratio, 1) * 100}%`;
}

async function requestPrediction() {
  if (buffer.length < SEQUENCE_LENGTH || isPredicting) return;
  isPredicting = true;
  statusOverlay.textContent = "Predicting...";
  predictButton.disabled = true;
  try {
    const response = await fetch(apiUrlInput.value.trim(), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sequence: buffer }),
    });
    if (!response.ok) {
      throw new Error(`Server responded with ${response.status}`);
    }
    const data = await response.json();
    predictionEl.textContent = data.prediction;
    confidenceEl.textContent = `${(data.confidence * 100).toFixed(1)}%`;
    dialogPrediction.textContent = data.prediction;
    dialogConfidence.textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
    dialog.showModal();
    statusOverlay.textContent = "Streaming";
  } catch (error) {
    log(`Prediction error: ${error.message}`);
    statusOverlay.textContent = "Error – check logs";
  } finally {
    isPredicting = false;
    predictButton.disabled = buffer.length !== SEQUENCE_LENGTH;
  }
}

function buildFrameVector(results) {
  const leftHand = zeroPadHand();
  const rightHand = zeroPadHand();

  if (!results.multiHandLandmarks || !results.multiHandedness) {
    return flattenHands(leftHand, rightHand);
  }

  for (let i = 0; i < results.multiHandLandmarks.length; i += 1) {
    const handedness = results.multiHandedness[i].label.toLowerCase();
    const coords = results.multiHandLandmarks[i].map((lm) => [lm.x, lm.y, lm.z]);
    const normalized = normalizeLandmarks(coords);
    if (handedness === "left") {
      for (let j = 0; j < normalized.length; j += 1) {
        leftHand[j] = normalized[j];
      }
    } else {
      for (let j = 0; j < normalized.length; j += 1) {
        rightHand[j] = normalized[j];
      }
    }
  }
  return flattenHands(leftHand, rightHand);
}

function drawResults(results) {
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
  if (results.multiHandLandmarks) {
    for (const landmarks of results.multiHandLandmarks) {
      drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, { color: "#22d3ee", lineWidth: 4 });
      drawLandmarks(canvasCtx, landmarks, { color: "#f472b6", lineWidth: 2, radius: 3 });
    }
  }
  canvasCtx.restore();
}

function onResults(results) {
  drawResults(results);
  if (isRecording) {
    const frameVec = buildFrameVector(results);
    rawFrames.push(frameVec);
  }
}

function startPipeline() {
  rawFrames.length = 0;
  buffer.length = 0;
  predictionEl.textContent = "–";
  confidenceEl.textContent = "–";
  statusOverlay.textContent = "Initializing...";
  progressFill.style.width = "0%";
  predictButton.disabled = true;
  recordButton.disabled = false;
  isRecording = false;
  if (recordAnimationId) {
    cancelAnimationFrame(recordAnimationId);
    recordAnimationId = null;
  }

  const hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4/${file}`,
  });
  hands.setOptions({
    maxNumHands: 2,
    modelComplexity: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.7,
  });
  hands.onResults(onResults);

  const camera = new Camera(videoElement, {
    onFrame: async () => {
      await hands.send({ image: videoElement });
    },
    width: 640,
    height: 480,
  });
  camera.start();
  log("Camera started and MediaPipe Hands initialized.");
  statusOverlay.textContent = "Streaming";
}

function resampleFrames(frames, targetLength) {
  if (!frames.length) return [];
  const indices = Array.from(
    { length: targetLength },
    (_, i) => Math.floor((i / (targetLength - 1)) * (frames.length - 1))
  );
  return indices.map((idx) => frames[idx]);
}

function finishRecording() {
  isRecording = false;
  recordAnimationId = null;
  recordButton.disabled = false;
  if (rawFrames.length < SEQUENCE_LENGTH) {
    statusOverlay.textContent = "Not enough frames collected. Try again.";
    rawFrames.length = 0;
    progressFill.style.width = "0%";
    return;
  }
  const resampled = resampleFrames(rawFrames, SEQUENCE_LENGTH);
  buffer.splice(0, buffer.length, ...resampled);
  rawFrames.length = 0;
  predictButton.disabled = false;
  statusOverlay.textContent = "Frames ready – click Show Prediction";
}

recordButton.addEventListener("click", () => {
  rawFrames.length = 0;
  buffer.length = 0;
  predictButton.disabled = true;
  isRecording = true;
  recordButton.disabled = true;
  progressFill.style.width = "0%";
  statusOverlay.textContent = "Recording...";

  const startTime = performance.now();
  const step = () => {
    if (!isRecording) return;
    const elapsed = performance.now() - startTime;
    updateProgress(elapsed / RECORD_DURATION_MS);
    if (elapsed >= RECORD_DURATION_MS) {
      finishRecording();
    } else {
      recordAnimationId = requestAnimationFrame(step);
    }
  };
  recordAnimationId = requestAnimationFrame(step);
});

predictButton.addEventListener("click", () => {
  if (buffer.length === SEQUENCE_LENGTH) {
    requestPrediction();
  }
});

resetButton.addEventListener("click", () => {
  rawFrames.length = 0;
  buffer.length = 0;
  progressFill.style.width = "0%";
  predictButton.disabled = true;
  recordButton.disabled = false;
  isRecording = false;
  if (recordAnimationId) {
    cancelAnimationFrame(recordAnimationId);
    recordAnimationId = null;
  }
  statusOverlay.textContent = "Streaming";
});

dialogClose.addEventListener("click", () => {
  dialog.close();
});

startPipeline();
