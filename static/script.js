const $ = (id) => document.getElementById(id);
const camBtn = $("camBtn"), toast = $("toast");
let running = false;

function showError(msg) {
  toast.textContent = msg;
  toast.classList.remove("hidden");
  setTimeout(() => toast.classList.add("hidden"), 5000);
}

function setUI(on) {
  running = on;
  camBtn.innerHTML = on ? "&#9632; Stop Camera" : "&#127909; Start Camera";
  camBtn.classList.toggle("stop", on);
  $("status").textContent = on ? "Camera running" : "Ready";
  $("dot").classList.toggle("live", on);
}

camBtn.addEventListener("click", async () => {
  camBtn.disabled = true;
  try {
    const res = await fetch(running ? "/api/stop" : "/api/start", { method: "POST" });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || "Request failed");
    setUI(!running);
  } catch (e) {
    showError(e.message + " Check that no other app is using the camera.");
  } finally {
    camBtn.disabled = false;
  }
});

const keys = ["face", "hands", "pose", "mirror"];
function pushSettings() {
  const body = {};
  keys.forEach((k) => (body[k] = $(k).checked));
  fetch("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}
keys.forEach((k) => $(k).addEventListener("change", pushSettings));

async function poll() {
  try {
    const s = await (await fetch("/api/stats")).json();
    $("sFaces").textContent = s.faces;
    $("sHands").textContent = s.hands;
    $("sPose").textContent = s.pose;
    $("sFps").textContent = s.fps;
    $("fpsTop").textContent = s.fps;
    if (running !== s.running) setUI(s.running);
  } catch (_) { /* server not reachable */ }
}
setInterval(poll, 500);
pushSettings();
