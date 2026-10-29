const state = {
  stream: null,
  timer: null,
  busy: false,
};

const camera = document.querySelector("#camera");
const canvas = document.querySelector("#capture-canvas");
const startButton = document.querySelector("#start-button");
const stopButton = document.querySelector("#stop-button");
const resetButton = document.querySelector("#reset-button");
const placeholder = document.querySelector("#camera-placeholder");
const status = document.querySelector("#connection-status");
const cameraState = document.querySelector("#camera-state");
const notice = document.querySelector("#notice");
const errorMessage = document.querySelector("#error-message");

function setStatus(label, className) {
  status.className = `status-pill ${className}`;
  status.lastElementChild.textContent = label;
}

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
  setStatus("Connection error", "status-error");
}

function clearError() {
  errorMessage.hidden = true;
}

function updateResult(result) {
  document.querySelector("#recognized-text").textContent =
    result.current_text || "Make a steady sign to begin.";
  document.querySelector("#detected-sign").textContent =
    result.predicted_letter || "—";
  document.querySelector("#confidence-value").textContent =
    `${Math.round((result.confidence || 0) * 100)}%`;
  document.querySelector("#fps-value").textContent =
    `${Number(result.fps || 0).toFixed(0)} FPS`;
  document.querySelector("#stability-badge").textContent =
    result.is_stable ? "Hand stable" : "Finding stability";
}

async function sendFrame() {
  if (state.busy || !state.stream || !camera.videoWidth) return;
  state.busy = true;
  canvas.width = camera.videoWidth;
  canvas.height = camera.videoHeight;
  canvas.getContext("2d").drawImage(camera, 0, 0);
  try {
    const response = await fetch("/ws", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: canvas.toDataURL("image/jpeg", .78).split(",")[1] }),
    });
    if (!response.ok) throw new Error((await response.json()).detail || "Recognition request failed");
    updateResult(await response.json());
    clearError();
    setStatus("API connected", "status-connected");
  } catch (error) {
    showError(error.message);
  } finally {
    state.busy = false;
  }
}

async function startRecognition() {
  clearError();
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
    camera.srcObject = state.stream;
    placeholder.hidden = true;
    cameraState.textContent = "LIVE";
    startButton.disabled = true;
    stopButton.disabled = false;
    notice.hidden = true;
    setStatus("Camera active", "status-connected");
    state.timer = window.setInterval(sendFrame, 250);
  } catch (error) {
    showError("Camera access was denied or is unavailable. Check browser permissions.");
  }
}

function stopRecognition() {
  if (state.timer) window.clearInterval(state.timer);
  state.timer = null;
  if (state.stream) state.stream.getTracks().forEach((track) => track.stop());
  state.stream = null;
  camera.srcObject = null;
  placeholder.hidden = false;
  cameraState.textContent = "OFFLINE";
  startButton.disabled = false;
  stopButton.disabled = true;
  setStatus("Ready to connect", "status-idle");
}

async function resetRecognition() {
  try {
    const response = await fetch("/reset", { method: "POST" });
    if (!response.ok) throw new Error("Could not clear the message");
    updateResult({ current_text: "", confidence: 0, fps: 0, is_stable: false });
    clearError();
  } catch (error) {
    showError(error.message);
  }
}

startButton.addEventListener("click", startRecognition);
stopButton.addEventListener("click", stopRecognition);
resetButton.addEventListener("click", resetRecognition);
window.addEventListener("beforeunload", stopRecognition);
