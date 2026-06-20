"""
RL-Training — Spurhalten mit PPO (Stable-Baselines3)

Starten:
    python train.py

Ergebnis:
    results/best_model/      <- bestes Modell (nach Evaluation)
    results/checkpoints/     <- Zwischenstände alle 10.000 Schritte
    results/plots/update_NNN <- 5 PNG-Plots nach jedem N-ten Update
"""

import csv
import sys
from pathlib import Path

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from drivelab_env import DrivelabEnv
from plotter import make_plots


HERE    = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

# --- Hyperparameter -------------------------------------------------
TOTAL_TIMESTEPS  = 500_000
N_ENVS           = 4
EVAL_FREQ        = 10_000
CHECKPOINT_FREQ  = 10_000
EVAL_EPISODES    = 5
PLOT_EVERY_N_UPDATES = 1    # nach jedem PPO-Update plotten


TRAJ_HEADER = [
    "t", "x", "y", "psi", "xdot", "ydot", "psidot",
    "rho_v", "rho_h", "Fvx", "Fvy", "Fhx", "Fhy", "dist",
    "s", "e_y", "e_psi", "distLeft", "distRight", "offRoad",
]


class EvalAndPlotCallback(BaseCallback):
    """
    Nach jedem PLOT_EVERY_N_UPDATES PPO-Updates:
    1. Eine deterministische Eval-Episode fahren
    2. Trajektorie als trajectory.csv speichern
    3. 5 Plots als PNG in results/plots/update_NNN/ speichern
    """

    def __init__(self, plot_every: int = PLOT_EVERY_N_UPDATES, verbose: int = 1):
        super().__init__(verbose)
        self._update_count = 0
        self._plot_every   = plot_every

    def _on_rollout_end(self) -> None:
        self._update_count += 1
        if self._update_count % self._plot_every != 0:
            return

        update_str = f"update_{self._update_count:04d}"
        plot_dir   = RESULTS / "plots" / update_str
        traj_path  = RESULTS / "trajectory.csv"
        track_path = RESULTS / "track.csv"

        if self.verbose:
            print(f"\n[EvalAndPlot] Update {self._update_count} — fahre Eval-Episode ...")

        # --- Eval-Episode -------------------------------------------
        eval_env = DrivelabEnv()
        obs_raw, _ = eval_env.reset()
        rows = []
        done = False

        while not done:
            # Obs normalisieren mit laufenden Statistiken des Training-Envs
            obs_norm = self.training_env.normalize_obs(
                obs_raw[np.newaxis, :]
            ).flatten()
            action, _ = self.model.predict(obs_norm, deterministic=True)
            obs_raw, _, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated

            state = info.get("state", [0.0] * 13)
            # state = [X, Y, PSI, XDOT, YDOT, PSIDOT, RHO_V, RHO_H, FVX, FVY, FHX, FHY, DIST]
            rows.append([
                f"{info.get('t', 0.0):.5f}",
                *[f"{v:.6g}" for v in state],       # x,y,psi,xdot,ydot,psidot,rho_v,rho_h,Fvx,Fvy,Fhx,Fhy,dist
                f"{info.get('s', 0.0):.6g}",
                f"{obs_raw[0]:.6g}",                # e_y
                f"{obs_raw[1]:.6g}",                # e_psi
                f"{obs_raw[3]:.6g}",                # distLeft
                f"{obs_raw[4]:.6g}",                # distRight
                int(info.get("offRoad", False)),
            ])

        eval_env.close()

        # --- Trajektorie speichern ----------------------------------
        with open(traj_path, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(TRAJ_HEADER)
            w.writerows(rows)

        steps = self._update_count * 2048 * N_ENVS
        if self.verbose:
            print(f"           {len(rows)} Schritte, Plots -> {plot_dir}")

        # --- Plots speichern ----------------------------------------
        try:
            make_plots(
                track_path,
                traj_path,
                title_suffix=f"  [Update {self._update_count}, ~{steps:,} steps]",
                save_dir=plot_dir,
                show=False,
            )
        except Exception as exc:
            print(f"[EvalAndPlot] Warnung: Plot fehlgeschlagen: {exc}")

    def _on_step(self) -> bool:
        return True


def make_env():
    return DrivelabEnv()


def main():
    print("=== DriveLab RL-Training (PPO) ===")
    print(f"  Gesamt-Schritte : {TOTAL_TIMESTEPS:,}")
    print(f"  Parallele Envs  : {N_ENVS}")
    print(f"  Plot alle       : {PLOT_EVERY_N_UPDATES} Updates")
    print()

    vec_train = make_vec_env(make_env, n_envs=N_ENVS)
    vec_train = VecNormalize(vec_train, norm_obs=True, norm_reward=True, clip_obs=10.0)

    vec_eval = make_vec_env(make_env, n_envs=1)
    vec_eval = VecNormalize(vec_eval, norm_obs=True, norm_reward=False,
                            clip_obs=10.0, training=False)

    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=str(RESULTS / "checkpoints"),
        name_prefix="ppo_drivelab",
        verbose=1,
    )
    eval_cb = EvalCallback(
        eval_env=vec_eval,
        best_model_save_path=str(RESULTS / "best_model"),
        log_path=str(RESULTS / "eval_logs"),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
        verbose=1,
    )
    plot_cb = EvalAndPlotCallback(plot_every=PLOT_EVERY_N_UPDATES, verbose=1)

    callbacks = CallbackList([checkpoint_cb, eval_cb, plot_cb])

    model = PPO(
        policy="MlpPolicy",
        env=vec_train,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
    )

    print("Starte Training...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)

    model_path = RESULTS / "ppo_drivelab_final"
    model.save(str(model_path))
    vec_train.save(str(RESULTS / "vecnormalize.pkl"))
    print(f"\nTraining abgeschlossen.")
    print(f"  Modell     -> {model_path}.zip")
    print(f"  Normierung -> {RESULTS / 'vecnormalize.pkl'}")


if __name__ == "__main__":
    main()
