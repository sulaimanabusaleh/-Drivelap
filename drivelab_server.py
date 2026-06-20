"""
drivelab_server.py — läuft in Python 3.13 (base), hält die drivelab-Simulation
und kommuniziert mit dem gymnasium-Wrapper via stdin/stdout (JSON-Protokoll).

Start: NICHT direkt aufrufen — wird von drivelab_env.py als Subprocess gestartet.

Protokoll (newline-delimited JSON):
  EINGEHEND (von env):
    {"cmd": "reset"}
    {"cmd": "step", "lenkwinkel": 0.123}
    {"cmd": "quit"}

  AUSGEHEND (an env):
    {"status": "ok", "obs": {e_y, e_psi, curvature, distLeft, distRight, offRoad}}
    {"status": "ok", "obs": {...}, "reward": 0.0, "done": false, "info": {...}}
    {"status": "error", "msg": "..."}
"""

import sys
import os
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

mingw_path = r"C:\Program Files\mingw64\bin"
if os.path.isdir(mingw_path):
    os.add_dll_directory(mingw_path)

import drivelab

# Reward-Gewichte (müssen identisch mit drivelab_env.py sein)
W_LATERAL    = 2.0
W_HEADING    = 0.5
CRASH_REWARD = -10.0

CONFIG = str(HERE / "config.json")
os.chdir(HERE / "results")
sim = drivelab.make_env(CONFIG)


def obs_to_dict(obs) -> dict:
    return {
        "e_y":       obs.e_y,
        "e_psi":     obs.e_psi,
        "curvature": obs.curvature,
        "distLeft":  obs.distLeft,
        "distRight": obs.distRight,
        "offRoad":   bool(obs.offRoad),
    }


def compute_reward(obs, terminated: bool) -> float:
    if terminated and obs.offRoad:
        return CRASH_REWARD
    r = 1.0
    r -= W_LATERAL * obs.e_y   ** 2
    r -= W_HEADING * obs.e_psi ** 2
    return r


# Signalisiere dem Elternprozess, dass wir bereit sind
sys.stdout.write(json.dumps({"status": "ready"}) + "\n")
sys.stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
        cmd = msg.get("cmd")

        if cmd == "reset":
            obs = sim.reset()
            reply = {"status": "ok", "obs": obs_to_dict(obs)}

        elif cmd == "step":
            lenkwinkel = float(msg["lenkwinkel"])
            act = drivelab.Action(lenkwinkel=lenkwinkel, gas=0.0, bremse=0.0)
            result = sim.step(act)
            obs = result.obs
            terminated = bool(result.done)
            reward = compute_reward(obs, terminated)
            state = list(sim.current_state())
            reply = {
                "status": "ok",
                "obs":    obs_to_dict(obs),
                "reward": reward,
                "done":   terminated,
                "info": {
                    "t":       sim.time(),
                    "s":       obs.s,
                    "e_y":     obs.e_y,
                    "offRoad": bool(obs.offRoad),
                    "state":   state,
                },
            }

        elif cmd == "quit":
            break

        else:
            reply = {"status": "error", "msg": f"Unbekannter Befehl: {cmd}"}

    except Exception as exc:
        reply = {"status": "error", "msg": str(exc)}

    sys.stdout.write(json.dumps(reply) + "\n")
    sys.stdout.flush()
