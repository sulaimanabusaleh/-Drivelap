"""
RL-Training — Spurhalten mit PPO (Stable-Baselines3)

Starten:
    python train.py                  <- speichert in results/run_1/ (oder naechste freie Nummer)
    python train.py training_1       <- speichert in results/training_1/
    python train.py mit_lanewidth    <- speichert in results/mit_lanewidth/

Ergebnis (im jeweiligen Ordner):
    results/<name>/best_model/       <- bestes Modell (nach Evaluation)
    results/<name>/checkpoints/      <- Zwischenstände alle 10.000 Schritte
    results/<name>/plots/update_NNN  <- 5 PNG-Plots nach jedem N-ten Update
    results/<name>/vecnormalize.pkl  <- Normalisierungs-Statistiken
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
N_ENVS               = 4
EVAL_FREQ            = 10_000
CHECKPOINT_FREQ      = 10_000
EVAL_EPISODES        = 5
PLOT_EVERY_N_UPDATES = 1

# --- Curriculum: Straße → Schritte ----------------------------------
CURRICULUM = [
    ("./data/roads/newRoad.json",              250_000),
    ("./data/roads/einfacher_Rundkurs.json",   250_000),
    ("./data/roads/road3_kurvig.json",          250_000),
    ("./data/roads/road4_komplex.json",         250_000),
]
TOTAL_TIMESTEPS = sum(steps for _, steps in CURRICULUM)  # 1_000_000


TRAJ_HEADER = [
    "t", "x", "y", "psi", "xdot", "ydot", "psidot",
    "rho_v", "rho_h", "Fvx", "Fvy", "Fhx", "Fhy", "dist",
    "s", "e_y", "e_psi", "distLeft", "distRight", "offRoad",
]


def next_run_name() -> str:
    """Gibt run_1, run_2, ... zurück — nächste freie Nummer."""
    i = 1
    while (RESULTS / f"run_{i}").exists():
        i += 1
    return f"run_{i}"


class EvalAndPlotCallback(BaseCallback):
    def __init__(self, run_dir: Path, road: str = None, plot_every: int = PLOT_EVERY_N_UPDATES, verbose: int = 1):
        super().__init__(verbose)
        self._update_count = 0
        self._plot_every   = plot_every
        self._run_dir      = run_dir
        self.road          = road  # aktuelle Straße — wird zwischen Phasen gesetzt

    def _on_rollout_end(self) -> None:
        self._update_count += 1
        # vecnormalize nach jedem Update speichern — damit eval.py immer funktioniert
        self.training_env.save(str(self._run_dir / "vecnormalize.pkl"))
        if self._update_count % self._plot_every != 0:
            return

        update_str = f"update_{self._update_count:04d}"
        plot_dir   = self._run_dir / "plots" / update_str
        traj_path  = self._run_dir / "trajectory.csv"
        track_path = RESULTS / "track.csv"

        if self.verbose:
            road_label = self.road or "zufällig"
            print(f"\n[EvalAndPlot] Update {self._update_count} — Straße: {road_label}")

        eval_env = DrivelabEnv(road=self.road)
        obs_raw, _ = eval_env.reset()
        rows = []
        done = False

        while not done:
            obs_norm = self.training_env.normalize_obs(
                obs_raw[np.newaxis, :]
            ).flatten()
            action, _ = self.model.predict(obs_norm, deterministic=True)
            obs_raw, _, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated

            state = info.get("state", [0.0] * 13)
            rows.append([
                f"{info.get('t', 0.0):.5f}",
                *[f"{v:.6g}" for v in state],
                f"{info.get('s', 0.0):.6g}",
                f"{obs_raw[0]:.6g}",
                f"{obs_raw[1]:.6g}",
                f"{obs_raw[3]:.6g}",
                f"{obs_raw[4]:.6g}",
                int(info.get("offRoad", False)),
            ])

        eval_env.close()

        with open(traj_path, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(TRAJ_HEADER)
            w.writerows(rows)

        steps = self._update_count * 2048 * N_ENVS
        if self.verbose:
            print(f"           {len(rows)} Schritte, Plots -> {plot_dir}")

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


def make_env(road: str = None):
    return DrivelabEnv(road=road)


def main():
    # --- Run-Name aus Argument oder automatisch ---------------------
    if len(sys.argv) > 1:
        run_name = sys.argv[1]
    else:
        run_name = next_run_name()

    run_dir = RESULTS / run_name
    if run_dir.exists():
        print(f"Warnung: Ordner '{run_dir}' existiert bereits — Dateien werden überschrieben.")
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=== DriveLab RL-Training (PPO) — Curriculum Learning ===")
    print(f"  Run-Name        : {run_name}")
    print(f"  Ergebnisse in   : {run_dir}")
    print(f"  Gesamt-Schritte : {TOTAL_TIMESTEPS:,}")
    print(f"  Parallele Envs  : {N_ENVS}")
    print(f"  Curriculum:")
    for road, steps in CURRICULUM:
        print(f"    {steps:>9,} Steps → {road}")
    print()

    # --- Phase 1: Modell erstellen mit erster Straße ----------------
    road0, steps0 = CURRICULUM[0]
    vec_train = make_vec_env(lambda: make_env(road0), n_envs=N_ENVS)
    vec_train = VecNormalize(vec_train, norm_obs=True, norm_reward=True, clip_obs=10.0)

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
        tensorboard_log=str(run_dir / "tensorboard"),
    )

    # --- Curriculum-Schleife ----------------------------------------
    for phase_idx, (road, steps) in enumerate(CURRICULUM):
        phase_num = phase_idx + 1
        print(f"\n{'='*55}")
        print(f"  PHASE {phase_num}/4 — {steps:,} Steps auf: {road}")
        print(f"{'='*55}\n")

        if phase_idx > 0:
            # Neue Umgebung mit neuer Straße, VecNormalize-Stats übernehmen
            vec_train_new = make_vec_env(lambda r=road: make_env(r), n_envs=N_ENVS)
            vecnorm_path  = str(run_dir / "vecnormalize.pkl")
            vec_train     = VecNormalize.load(vecnorm_path, vec_train_new)
            vec_train.training   = True
            vec_train.norm_reward = True
            model.set_env(vec_train)

        checkpoint_cb = CheckpointCallback(
            save_freq=CHECKPOINT_FREQ,
            save_path=str(run_dir / "checkpoints"),
            name_prefix=f"ppo_phase{phase_num}",
            verbose=1,
        )
        vec_eval = make_vec_env(lambda r=road: make_env(r), n_envs=1)
        vec_eval = VecNormalize(vec_eval, norm_obs=True, norm_reward=False,
                                clip_obs=10.0, training=False)
        eval_cb = EvalCallback(
            eval_env=vec_eval,
            best_model_save_path=str(run_dir / "best_model"),
            log_path=str(run_dir / "eval_logs"),
            eval_freq=EVAL_FREQ,
            n_eval_episodes=EVAL_EPISODES,
            deterministic=True,
            verbose=1,
        )
        plot_cb = EvalAndPlotCallback(run_dir=run_dir, road=road, plot_every=PLOT_EVERY_N_UPDATES, verbose=1)
        callbacks = CallbackList([checkpoint_cb, eval_cb, plot_cb])

        model.learn(
            total_timesteps=steps,
            callback=callbacks,
            reset_num_timesteps=False,  # Schritte addieren, nicht neu starten
        )

        # VecNormalize nach jeder Phase speichern
        vec_train.save(str(run_dir / "vecnormalize.pkl"))
        model.save(str(run_dir / f"ppo_phase{phase_num}_end"))
        print(f"\n  Phase {phase_num} abgeschlossen — Modell gespeichert.")

    model.save(str(run_dir / "ppo_drivelab_final"))
    vec_train.save(str(run_dir / "vecnormalize.pkl"))
    print(f"\nTraining abgeschlossen.")
    print(f"  Modell     -> {run_dir / 'ppo_drivelab_final.zip'}")
    print(f"  Normierung -> {run_dir / 'vecnormalize.pkl'}")


if __name__ == "__main__":
    main()
