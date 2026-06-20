"""
DrivelabEnv — Gymnasium-Wrapper um die drivelab-Simulation.

Architektur (Zwei-Prozess-Bridge):
  drivelab_server.py  läuft in Python 3.13 (base) — lädt drivelab.cp313
  drivelab_env.py     läuft in Python 3.12 (py312) — hat gymnasium + SB3
  Kommunikation: newline-delimited JSON über stdin/stdout

Ziel:     Spurhalten (Lane Keeping)
Action:   lenkwinkel [rad]  (1D, kontinuierlich, [-0.5, 0.5])
Obs:      [e_y, e_psi, curvature, distLeft, distRight]  (5D)
Reward:   +1/Schritt - 2*e_y² - 0.5*e_psi²  (Crash: -10)
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

HERE = Path(__file__).resolve().parent

# Pfad zum Python 3.13 (base) Interpreter, der drivelab laden kann
PYTHON_313 = r"C:\Users\WZLFPYG\AppData\Local\anaconda3\python.exe"
SERVER_SCRIPT = str(HERE / "drivelab_server.py")

# --- Grenzen für den Observation Space ---
OBS_LOW  = np.array([-5.0,  -1.5,  -0.2,   0.0,   0.0,   0.0,   0.0,  -5.0], dtype=np.float32)
OBS_HIGH = np.array([ 5.0,   1.5,   0.2,  10.0,  10.0,  10.0,  50.0,   5.0], dtype=np.float32)
#                    e_y    e_psi  curv  distL  distR  laneW    v   psidot

LENKWINKEL_MAX = 0.5


class DrivelabEnv(gym.Env):
    """
    Gymnasium-Umgebung für das DriveLab-Spurhalte-Problem.
    Kommuniziert mit drivelab_server.py (Python 3.13) via subprocess.
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # Action Space: lenkwinkel in [-0.5, 0.5] rad
        self.action_space = spaces.Box(
            low=np.array([-LENKWINKEL_MAX], dtype=np.float32),
            high=np.array([ LENKWINKEL_MAX], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation Space: [e_y, e_psi, curvature, distLeft, distRight]
        self.observation_space = spaces.Box(
            low=OBS_LOW, high=OBS_HIGH, dtype=np.float32
        )

        self._proc = None
        self._start_server()

    # ------------------------------------------------------------------
    def _start_server(self):
        """Startet drivelab_server.py als Subprocess."""
        if self._proc is not None and self._proc.poll() is None:
            return  # läuft bereits

        self._proc = subprocess.Popen(
            [PYTHON_313, SERVER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,  # zeilenweise gepuffert
        )
        # Warten bis Server "ready" meldet
        ready_line = self._proc.stdout.readline()
        msg = json.loads(ready_line)
        if msg.get("status") != "ready":
            raise RuntimeError(f"Server nicht bereit: {msg}")

    def _send(self, payload: dict) -> dict:
        """Sendet ein JSON-Kommando und liest die Antwort."""
        line = json.dumps(payload) + "\n"
        self._proc.stdin.write(line)
        self._proc.stdin.flush()
        reply = self._proc.stdout.readline()
        if not reply:
            raise RuntimeError("Server hat die Verbindung getrennt.")
        result = json.loads(reply)
        if result.get("status") == "error":
            raise RuntimeError(f"Server-Fehler: {result['msg']}")
        return result

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        reply = self._send({"cmd": "reset"})
        return self._obs_to_array(reply["obs"]), {}

    def step(self, action):
        lenkwinkel = float(np.clip(action[0], -LENKWINKEL_MAX, LENKWINKEL_MAX))
        reply = self._send({"cmd": "step", "lenkwinkel": lenkwinkel})
        obs_arr    = self._obs_to_array(reply["obs"])
        reward     = float(reply["reward"])
        terminated = bool(reply["done"])
        truncated  = False
        info = reply.get("info", {})
        return obs_arr, reward, terminated, truncated, info

    def close(self):
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._send({"cmd": "quit"})
            except Exception:
                pass
            self._proc.terminate()
            self._proc = None

    # ------------------------------------------------------------------
    def _obs_to_array(self, obs: dict) -> np.ndarray:
        return np.array(
            [obs["e_y"], obs["e_psi"], obs["curvature"],
             obs["distLeft"], obs["distRight"], obs["laneWidth"],
             obs["v"], obs["psidot"]],
            dtype=np.float32,
        )
