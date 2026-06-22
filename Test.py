
import sys
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
             
sys.path.insert(0, str(HERE))

import os
mingw_path = r"C:\Program Files\mingw64\bin"
if os.path.isdir(mingw_path):
    os.add_dll_directory(mingw_path)
import drivelab


CONFIG = HERE / "config.json"

os.chdir(RESULTS)
env = drivelab.make_env(str(CONFIG))

HEADER = [
    "t", "x", "y", "psi", "xdot", "ydot", "psidot",
    "rho_v", "rho_h", "Fvx", "Fvy", "Fhx", "Fhy", "dist",
    "s", "e_y", "e_psi", "distLeft", "distRight", "offRoad",
]

env.reset()

with open(RESULTS / "trajectory.csv", "w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(HEADER)

    done = False
    step = 0
    max_steps = 100_000      
    while not done and step < max_steps:
        a = drivelab.Action(lenkwinkel=0.0, gas=0.0, bremse=0.0)
        r = env.step(a)
        obs = r.obs

        (x, y, psi, xdot, ydot, psidot,
         rho_v, rho_h, Fvx, Fvy, Fhx, Fhy, dist) = env.current_state()

        w.writerow([
            f"{env.time():.5f}", x, y, psi, xdot, ydot, psidot,
            rho_v, rho_h, Fvx, Fvy, Fhx, Fhy, dist,
            obs.s, obs.e_y, obs.e_psi, obs.distLeft, obs.distRight, int(obs.offRoad),
        ])

        done = r.done
        step += 1

print(f"Fertig: {step} Schritte, t = {env.time():.2f} s")
print(f"  -> {RESULTS / 'trajectory.csv'}")
print(f"  -> {RESULTS / 'track.csv'}")