import { spawn } from "node:child_process";
import { setTimeout as delay } from "node:timers/promises";

const VIEWER_ORIGIN = "http://127.0.0.1:4173";
const VIEWER_INDEX = `${VIEWER_ORIGIN}/viewer/index.html`;

export async function ensureViewerServer({ timeoutMs = 5000 } = {}) {
  if (await viewerServerReady()) {
    return null;
  }

  const server = spawn("python3", ["-m", "http.server", "4173"], {
    cwd: process.cwd(),
    stdio: ["ignore", "ignore", "pipe"],
  });
  let stderr = "";
  let exited = false;
  server.stderr.on("data", (chunk) => {
    stderr += chunk.toString();
  });
  server.once("exit", () => {
    exited = true;
  });

  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await viewerServerReady()) {
      return server;
    }
    if (exited) {
      break;
    }
    await delay(100);
  }

  await stopViewerServer(server);
  const detail = stderr.trim() ? ` ${stderr.trim()}` : "";
  throw new Error(`Viewer server did not start on ${VIEWER_ORIGIN}.${detail}`);
}

export async function stopViewerServer(server) {
  if (!server || server.exitCode !== null || server.signalCode !== null) {
    return;
  }
  server.kill("SIGTERM");
  await Promise.race([
    new Promise((resolve) => server.once("exit", resolve)),
    delay(1000).then(() => {
      if (server.exitCode === null && server.signalCode === null) {
        server.kill("SIGKILL");
      }
    }),
  ]);
}

async function viewerServerReady() {
  try {
    const response = await fetch(VIEWER_INDEX);
    return response.ok;
  } catch {
    return false;
  }
}
