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
import random
import tempfile
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
W_STEER      = 0.5    # Run C: sanftes Lenken
CRASH_REWARD = -10.0

# Reward-Modus: "A" = v_target fix, "B" = Fortschritt, "C" = Fortschritt + Lenkstrafe
REWARD_MODE  = "B"
V_TARGET     = 10.0   # nur für Modus A

# Alle verfügbaren Straßen
ROADS = [
    "./data/roads/newRoad.json",
    "./data/roads/einfacher_Rundkurs.json",
    "./data/roads/road3_kurvig.json",
    "./data/roads/road4_komplex.json",
]

def make_config(road: str) -> str:
    """Erstellt eine temporäre config.json mit der gewählten Straße."""
    base = json.loads((HERE / "config.json").read_text())
    base["road"] = road
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir=HERE
    )
    json.dump(base, tmp)
    tmp.close()
    return tmp.name

def road_length(road: str) -> float:
    """Berechnet die Gesamtlänge der Straße aus der JSON."""
    data = json.loads((HERE / road.lstrip("./")).read_text())
    return sum(seg["length"] for seg in data["segments"])

os.chdir(HERE / "results")
# Straße aus Argument oder zufällig
if len(sys.argv) > 1:
    road_name = sys.argv[1]
    matched = next((r for r in ROADS if road_name in r), None)
    road = matched if matched else road_name  # unbekannte Straße direkt verwenden
else:
    road = random.choice(ROADS)
config_path = make_config(road)
sim = drivelab.make_env(config_path)
os.unlink(config_path)
ROAD_LENGTH = road_length(road)


def obs_to_dict(obs, state: list) -> dict:
    xdot   = state[3]
    ydot   = state[4]
    psidot = state[5]
    v = (xdot**2 + ydot**2) ** 0.5
    return {
        "e_y":       obs.e_y,
        "e_psi":     obs.e_psi,
        "curvature": obs.curvature,
        "laneWidth": obs.laneWidth,
        "distLeft":  obs.distLeft,
        "distRight": obs.distRight,
        "v":         v,
        "psidot":    psidot,
        "offRoad":   bool(obs.offRoad),
    }


def compute_reward(obs, state: list, terminated: bool,
                   s_prev: float = 0.0, lenkwinkel_prev: float = 0.0,
                   lenkwinkel: float = 0.0) -> float:
    if terminated and obs.offRoad:
        return CRASH_REWARD

    xdot = state[3]
    ydot = state[4]
    v    = (xdot**2 + ydot**2) ** 0.5

    if REWARD_MODE == "A":
        # --- Run A: feste Zielgeschwindigkeit ---
        r  = 1.0
        r -= W_LATERAL * obs.e_y   ** 2
        r -= W_HEADING * obs.e_psi ** 2
        r -= 0.3       * (v - V_TARGET) ** 2

    elif REWARD_MODE == "B":
        # --- Run B: Fortschritt (zeitoptimal) ---
        fortschritt = obs.s - s_prev
        r  = fortschritt
        r -= W_LATERAL * obs.e_y   ** 2
        r -= W_HEADING * obs.e_psi ** 2

    else:  # "C"
        # --- Run C: Fortschritt + sanftes Lenken ---
        fortschritt  = obs.s - s_prev
        lenkänderung = abs(lenkwinkel - lenkwinkel_prev)
        r  = fortschritt
        r -= W_LATERAL * obs.e_y   ** 2
        r -= W_HEADING * obs.e_psi ** 2
        r -= W_STEER   * lenkänderung

    return r


# Zustand zwischen Schritten merken (für Fortschritts-Reward)
s_prev         = 0.0
lenkwinkel_prev = 0.0

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
            s_prev          = obs.s
            lenkwinkel_prev = 0.0
            reply = {"status": "ok", "obs": obs_to_dict(obs, list(sim.current_state()))}

        elif cmd == "step":
            lenkwinkel = float(msg["lenkwinkel"])
            gas        = float(msg.get("gas",    0.0))
            bremse     = float(msg.get("bremse", 0.0))
            act = drivelab.Action(lenkwinkel=lenkwinkel, gas=gas, bremse=bremse)
            result = sim.step(act)
            obs = result.obs
            terminated = bool(result.done) or obs.s >= ROAD_LENGTH
            state = list(sim.current_state())
            reward = compute_reward(obs, state, terminated,
                                    s_prev=s_prev,
                                    lenkwinkel_prev=lenkwinkel_prev,
                                    lenkwinkel=lenkwinkel)
            s_prev          = obs.s
            lenkwinkel_prev = lenkwinkel
            reply = {
                "status": "ok",
                "obs":    obs_to_dict(obs, state),
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
