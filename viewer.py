"""
viewer.py — Live-Viewer fuer die Trainings-Plots.

Laeuft parallel zum Training und zeigt automatisch die neuesten Plots
sobald ein neues Update-Verzeichnis auftaucht.

Starten (in separatem Terminal):
    python viewer.py

Beenden: Fenster schliessen oder Ctrl+C im Terminal.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

HERE      = Path(__file__).resolve().parent
PLOTS_DIR = HERE / "results" / "plots"

PLOT_FILES = [
    "01_trajektorie.png",
    "02_querabweichung.png",
    "03_heading_fehler.png",
    "04_randabstaende.png",
    "05_geschwindigkeit.png",
]

CHECK_INTERVAL = 5   # Sekunden zwischen Prüfungen auf neues Update

plt.ion()  # interaktiver Modus — Fenster bleibt offen während Code weiterläuft


def get_latest_update_dir():
    """Gibt das neueste update_XXXX-Verzeichnis zurück (oder None)."""
    if not PLOTS_DIR.exists():
        return None
    dirs = sorted(
        [d for d in PLOTS_DIR.iterdir() if d.is_dir() and d.name.startswith("update_")]
    )
    return dirs[-1] if dirs else None


TITLES = [
    "Trajektorie",
    "Querabweichung",
    "Heading-Fehler",
    "Randabstaende",
    "Geschwindigkeit",
]

# Fensterpositionen (x, y) in Pixeln — nebeneinander/untereinander
POSITIONS = [
    (0,   0),
    (640, 0),
    (0,   480),
    (640, 480),
    (320, 240),
]


def show_plots(update_dir: Path):
    """Öffnet jedes PNG in einem eigenen Fenster."""
    figs = []
    for i, fname in enumerate(PLOT_FILES):
        p = update_dir / fname
        if not p.exists():
            continue

        img = mpimg.imread(str(p))
        fig, ax = plt.subplots(figsize=(10, 5), num=f"Plot {i+1} — {TITLES[i]}")
        fig.canvas.manager.set_window_title(
            f"{TITLES[i]}  [{update_dir.name}]"
        )
        ax.imshow(img)
        ax.axis("off")
        plt.tight_layout()

        # Fenster positionieren
        try:
            x, y = POSITIONS[i]
            fig.canvas.manager.window.wm_geometry(f"+{x}+{y}")
        except Exception:
            pass

        plt.draw()
        figs.append(fig)

    plt.pause(0.1)
    return figs


def main():
    print("=== DriveLab Live-Viewer ===")
    print(f"Beobachte: {PLOTS_DIR}")
    print(f"Prüfintervall: {CHECK_INTERVAL} s")
    print("Warte auf erstes Update ...\n")

    current_dir = None
    figs = []

    while True:
        latest = get_latest_update_dir()

        if latest and latest != current_dir:
            current_dir = latest
            print(f"Neues Update: {current_dir.name}")

            for f in figs:
                plt.close(f)

            figs = show_plots(current_dir) or []

        # plt.pause hält Fenster aktiv und gibt CPU frei
        plt.pause(CHECK_INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nViewer beendet.")
        plt.close("all")
