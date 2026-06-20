import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def make_plots(track_path, trajectory_path, title_suffix="", save_dir=None, show=True):
    """
    Erstellt alle 5 Plots.
    - save_dir: Ordner fuer PNG-Dateien (None = nicht speichern)
    - show:     plt.show() aufrufen (True beim interaktiven Aufruf)
    """
    save_dir = Path(save_dir) if save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        matplotlib.use("Agg")   # kein Fenster, nur Dateien

    df_track = pd.read_csv(track_path)
    df = pd.read_csv(trajectory_path, sep="\t")
    df["v"] = np.hypot(df["xdot"], df["ydot"])

    def _save_or_show(name):
        plt.tight_layout()
        if save_dir:
            plt.savefig(save_dir / f"{name}.png", dpi=120)
            plt.close()

    # Plot 1: Strecke + Fahrzeugtrajektorie
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.fill(
        list(df_track["lx"]) + list(df_track["rx"])[::-1],
        list(df_track["ly"]) + list(df_track["ry"])[::-1],
        color="0.85", zorder=0, label="Fahrbahn",
    )
    ax.plot(df_track["cx"], df_track["cy"], "--", color="0.4", lw=1.2, label="Mitte")
    ax.plot(df_track["lx"], df_track["ly"], color="C0", lw=1.5, label="links")
    ax.plot(df_track["rx"], df_track["ry"], color="C3", lw=1.5, label="rechts")
    ax.plot(df["x"], df["y"], color="C2", lw=2.0, label="Fahrzeug")
    ax.scatter(df["x"].iloc[0], df["y"].iloc[0], color="green", s=60, zorder=5, label="Start")
    ax.scatter(df["x"].iloc[-1], df["y"].iloc[-1], color="red", s=60, zorder=5, label="Ende")
    offroad = df["offRoad"].astype(bool)
    if offroad.any():
        ax.scatter(df.loc[offroad, "x"], df.loc[offroad, "y"],
                   color="red", s=12, zorder=6, label="Offroad")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Strecke und Fahrzeugtrajektorie{title_suffix}")
    _save_or_show("01_trajektorie")

    # Plot 2: Querabweichung
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["t"], df["e_y"], label="e_y")
    ax.axhline(0.0, color="0.4", lw=1)
    ax.grid(alpha=0.3)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("e_y [m]")
    ax.set_title(f"Lateraler Abstand zur Strassenmitte{title_suffix}")
    ax.legend()
    _save_or_show("02_querabweichung")

    # Plot 3: Heading-Fehler
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["t"], df["e_psi"], label="e_psi")
    ax.axhline(0.0, color="0.4", lw=1)
    ax.grid(alpha=0.3)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("e_psi [rad]")
    ax.set_title(f"Heading-Fehler Fahrzeug zu Strasse{title_suffix}")
    ax.legend()
    _save_or_show("03_heading_fehler")

    # Plot 4: Abstand zu Fahrbahnraendern
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["t"], df["distLeft"], label="Abstand links")
    ax.plot(df["t"], df["distRight"], label="Abstand rechts")
    ax.axhline(0.0, color="red", lw=1, linestyle="--", label="Grenze")
    ax.grid(alpha=0.3)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Abstand [m]")
    ax.set_title(f"Abstand zu Fahrbahnraendern{title_suffix}")
    ax.legend()
    _save_or_show("04_randabstaende")

    # Plot 5: Geschwindigkeit und Strecke
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(df["t"], df["v"], label="v")
    ax1.set_xlabel("t [s]")
    ax1.set_ylabel("v [m/s]")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(df["t"], df["dist"], linestyle="--", label="dist")
    ax2.set_ylabel("dist [m]")
    ax1.set_title(f"Geschwindigkeit und zurueckgelegte Strecke{title_suffix}")
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2)
    _save_or_show("05_geschwindigkeit")

    if show and not save_dir:
        plt.show()


# --- Direkt aufgerufen: interaktive Fenster anzeigen ----------------
if __name__ == "__main__":
    track_path      = RESULTS / "track.csv"
    trajectory_path = RESULTS / "trajectory.csv"

    missing = [p.name for p in (track_path, trajectory_path) if not p.exists()]
    if missing:
        sys.exit(
            f"Fehlende Datei(en) in {RESULTS}: {', '.join(missing)}\n"
            f"Bitte zuerst Test.py ausfuehren."
        )

    make_plots(track_path, trajectory_path, show=True)
