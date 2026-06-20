"""
eval.py — Bestes Modell laden und testen.

Starten:
    python eval.py                                         <- nimmt results/best_model/best_model.zip
    python eval.py results/checkpoints/ppo_drivelab_10000_steps.zip
    python eval.py C:/eigener/pfad/modell.zip
"""

import csv
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from drivelab_env import DrivelabEnv
from plotter import make_plots

HERE    = Path(__file__).resolve().parent
RESULTS = HERE / "results"

TRAJ_HEADER = [
    "t", "x", "y", "psi", "xdot", "ydot", "psidot",
    "rho_v", "rho_h", "Fvx", "Fvy", "Fhx", "Fhy", "dist",
    "s", "e_y", "e_psi", "distLeft", "distRight", "offRoad",
]


def main():
    # --- Pfad aus Argument oder Standard ---
    if len(sys.argv) > 1:
        best_model_path = Path(sys.argv[1])
    else:
        best_model_path = RESULTS / "best_model" / "best_model.zip"

    vecnorm_path = RESULTS / "vecnormalize.pkl"

    if not best_model_path.exists():
        sys.exit(
            f"Modell nicht gefunden: {best_model_path}\n"
            f"Benutze: python eval.py <pfad_zum_modell.zip>"
        )

    print("=== DriveLab Eval ===")
    print(f"  Modell: {best_model_path}")
    print()

    # --- Normierung laden (gleiche Statistiken wie beim Training) ---
    vec_env = make_vec_env(DrivelabEnv, n_envs=1)
    if vecnorm_path.exists():
        vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print(f"  Normierung geladen: {vecnorm_path}")
    else:
        print("  Warnung: vecnormalize.pkl nicht gefunden — keine Normierung!")

    # --- Modell laden ---
    model = PPO.load(str(best_model_path), env=vec_env)
    print("  Modell geladen.\n")

    # --- Episode fahren ---
    print("Fahre Eval-Episode ...")
    obs = vec_env.reset()
    done = False
    rows = []
    total_reward = 0.0
    step = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, info = vec_env.step(action)
        total_reward += float(reward[0])
        done = bool(terminated[0])
        step += 1

        state = info[0].get("state", [0.0] * 13)
        obs_raw = vec_env.get_original_obs()[0]

        rows.append([
            f"{info[0].get('t', 0.0):.5f}",
            *[f"{v:.6g}" for v in state],
            f"{info[0].get('s', 0.0):.6g}",
            f"{obs_raw[0]:.6g}",   # e_y
            f"{obs_raw[1]:.6g}",   # e_psi
            f"{obs_raw[3]:.6g}",   # distLeft
            f"{obs_raw[4]:.6g}",   # distRight
            int(info[0].get("offRoad", False)),
        ])

    vec_env.close()

    print(f"  Schritte       : {step}")
    print(f"  Gesamt-Reward  : {total_reward:.1f}")
    print(f"  offRoad        : {info[0].get('offRoad', False)}")
    print()

    # --- Trajektorie speichern ---
    traj_path  = RESULTS / "eval_trajectory.csv"
    track_path = RESULTS / "track.csv"

    with open(traj_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(TRAJ_HEADER)
        w.writerows(rows)

    print(f"  Trajektorie gespeichert: {traj_path}")

    # --- Plots anzeigen ---
    print("  Erstelle Plots ...")
    make_plots(
        track_path,
        traj_path,
        title_suffix=f"  [{best_model_path.stem}]",
        show=True,
    )


if __name__ == "__main__":
    main()
