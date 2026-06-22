"""
eval.py — Trainiertes Modell laden und testen.

Starten:
    python eval.py results/run_1/best_model/best_model.zip
    python eval.py results/run_1/checkpoints/ppo_drivelab_10000_steps.zip
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
    # --- Pfad hier ändern ---
    MODEL_PATH = "results/run_3/best_model/best_model.zip"
    # ------------------------

    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / MODEL_PATH
    road       = sys.argv[2] if len(sys.argv) > 2 else None

    if not model_path.exists():
        sys.exit(f"Modell nicht gefunden: {model_path}")

    run_dir      = model_path.parent.parent
    vecnorm_path = run_dir / "vecnormalize.pkl"

    print("=== DriveLab Eval ===")
    print(f"  Modell     : {model_path}")
    print(f"  Straße     : {road if road else 'zufällig'}")
    print(f"  Run-Ordner : {run_dir}")
    print()

    # --- Normierung laden ---
    vec_env = make_vec_env(lambda: DrivelabEnv(road=road), n_envs=1)
    if vecnorm_path.exists():
        vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print(f"  Normierung geladen: {vecnorm_path}")
    else:
        print("  Warnung: vecnormalize.pkl nicht gefunden — keine Normierung!")

    # --- Modell laden ---
    model = PPO.load(str(model_path), env=vec_env)
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
            f"{obs_raw[0]:.6g}",
            f"{obs_raw[1]:.6g}",
            f"{obs_raw[3]:.6g}",
            f"{obs_raw[4]:.6g}",
            int(info[0].get("offRoad", False)),
        ])

    vec_env.close()

    print(f"  Schritte       : {step}")
    print(f"  Gesamt-Reward  : {total_reward:.1f}")
    print(f"  offRoad        : {info[0].get('offRoad', False)}")
    print()

    # --- Trajektorie speichern ---
    traj_path  = run_dir / "eval_trajectory.csv"
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
        title_suffix=f"  [{model_path.stem}]",
        show=True,
    )


if __name__ == "__main__":
    main()
