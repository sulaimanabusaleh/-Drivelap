# -Drivelap

Wie es funktioniert (Schritt für Schritt)


┌─────────────────┐        JSON über stdin/stdout        ┌──────────────────────┐
│  drivelab_env.py│  ◄──────────────────────────────────►│ drivelab_server.py   │
│  (Python 3.12)  │                                      │ (Python 3.13)        │
│  KI / Gymnasium │                                      │ Physik-Simulation    │
└─────────────────┘                                      └──────────────────────┘


1. Die Simulation (drivelab_server.py) — läuft in Python 3.13

Enthält ein echtes Fahrzeugphysik-Modell (2000 kg Auto, Reifenmodell, Lenkung)
Simuliert: Position, Geschwindigkeit, Reifenkräfte

Gibt der KI 5 Informationen zurück:
1.e_y → wie weit das Auto seitlich von der Spurmitte abweicht
2.e_psi → wie schräg das Auto zur Straße steht
3.curvature → wie stark die Kurve ist
4,5.distLeft/distRight → Abstand zum linken/rechten Rand


2. Die KI-Umgebung (drivelab_env.py)

Verbindet die Simulation mit dem KI-Framework (Gymnasium)
Die KI darf nur lenken (Lenkwinkel von -0.5 bis +0.5 Rad)
Belohnungssystem:
+1 pro Schritt (für Überleben)
-2 × e_y² (Strafe für seitliche Abweichung)
-0.5 × e_psi² (Strafe für schräges Fahren)
-10 bei Unfall (von der Straße)


3. Das Training (train.py) — das Herzstück

Verwendet PPO (Proximal Policy Optimization) — einen modernen RL-Algorithmus
Trainiert 4 Autos gleichzeitig (parallel) für Geschwindigkeit
500.000 Schritte insgesamt
Speichert automatisch das beste Modell
Erstellt nach jedem Update 5 Analyse-Plots

4. Die Auswertung (eval.py)

Lädt das trainierte Modell
Fährt eine komplette Test-Runde
Zeigt alle Plots an

5. Die Plots (plotter.py) — 5 Grafiken:

Strecke + Fahrzeugweg (Draufsicht)
Seitliche Abweichung über Zeit
Heading-Fehler (Winkel zur Straße)
Abstände zu Fahrbahnrändern
Geschwindigkeit + zurückgelegte Strecke
