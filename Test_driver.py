
import sys
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent   
ROOT = HERE.parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

sys.path.insert(0, str(HERE))

import os
mingw_path = r"C:\Program Files\mingw64\bin"
if os.path.isdir(mingw_path):
    os.add_dll_directory(mingw_path)
import drivelab


os.chdir(RESULTS)
env = drivelab.make_env(str(HERE / "config.json"))
obs = env.reset()

HEADER = [
    "t", "x", "y", "psi", "xdot", "ydot", "psidot",
    "rho_v", "rho_h", "Fvx", "Fvy", "Fhx", "Fhy", "dist",
    "s", "e_y", "e_psi", "distLeft", "distRight", "offRoad",
]
# with open(RESULTS / "trajectory.csv", "w", newline="") as f:
#     w = csv.writer(f, delimiter="\t")
#     w.writerow(HEADER)'

for i in range(10000):
    x, y, psi, xdot, ydot, psidot, rho_v, rho_h, Fvx, Fvy, Fhx, Fhy, dist = env.current_state()

    # y = seitliche Position
    # psi = Gierwinkel (Heading)
    #Das ist ein P-Regler (simpelste Form): δ= −k⋅e
    #Zwei Fehler (e_y + e_psi) → PD-Regler: δ= − k ⋅ e_y − k ⋅ e_ψ 

    # PD-Spurhalte-Regler: bleibt auf der Strassenmitte
    k_y   = 0.15   # Gewichtung Querabweichung (e_y)
    k_psi = 0.8    # Gewichtung Winkelabweichung (e_psi)
    lenkwinkel = -k_y * obs.e_y - k_psi * obs.e_psi
    lenkwinkel = max(-0.5, min(0.5, lenkwinkel))  # begrenzen auf [-0.5, 0.5]

    a = drivelab.Action(lenkwinkel=lenkwinkel, gas=0.0, bremse=0.0)
    r = env.step(a)
    obs = r.obs

    # w.writerow([
    #     f"{env.time():.5f}", x, y, psi, xdot, ydot, psidot,
    #     rho_v, rho_h, Fvx, Fvy, Fhx, Fhy, dist,
    #     obs.s, obs.e_y, obs.e_psi, obs.distLeft, obs.distRight, int(obs.offRoad),
    # ])

    # if i % 100 == 0:
    #     print(f"=== t = {env.time():6.2f} s   (Schritt {i}) ===")
    #     print(f"  Straße:   s={obs.s:8.2f}  e_y={obs.e_y:+.3f}  e_psi={obs.e_psi:+.3f}  curv={obs.curvature:+.4f}")
    #     print(f"            lane={obs.laneWidth:.2f}  distL={obs.distLeft:.2f}  distR={obs.distRight:.2f}  offRoad={obs.offRoad}")
    #     print(f"  Position: x={x:8.2f}  y={y:8.2f}  psi={psi:+.3f}  zurückgelegt={dist:8.2f}")
    #     print(f"  Geschw.:  xdot={xdot:6.2f}  ydot={ydot:6.2f}  psidot={psidot:+.3f}  rho_v={rho_v:6.2f}  rho_h={rho_h:6.2f}")
    #     print(f"  Kräfte:   Fvx={Fvx:8.1f}  Fvy={Fvy:8.1f}  Fhx={Fhx:8.1f}  Fhy={Fhy:8.1f}")
    #     print(f"  Step:     reward={r.reward:.3f}  done={r.done}")
    if r.done:
        print("done bei t =", env.time())
        break

print(f"Gespeichert: {RESULTS / 'trajectory.csv'}")