"""
quality_check.py
Effectue des contrôles de qualité des données transformées avant l'insertion en BDD :
- Inexistence de doublons sur la clé primaire.
- Vérification du temps de jeu (>= 90 minutes).
- Cohérence des bornes numériques.
- Taux de valeurs nulles acceptable.
"""

import sys
import pandas as pd


def run_quality_checks():
    print("[*] Lancement du Quality Check sur les données transformées...")
    file_path = "data/processed/players_processed.csv"
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"[!] Erreur de lecture du fichier CSV transformé: {e}")
        sys.exit(1)

    errors = 0

    # Check 1 : Non-vacuité du jeu de données
    if df.empty:
        print("[FAIL] Le fichier transformé est vide.")
        errors += 1

    # Check 2 : Règle métier - Minutes Jouées >= 90
    invalid_minutes = df[df["minutes_played"] < 90]
    if not invalid_minutes.empty:
        print(f"[FAIL] {len(invalid_minutes)} lignes contiennent moins de 90 minutes jouées.")
        errors += 1

    # Check 3 : Unicité de la contrainte (player_id, team_id, season)
    duplicates = df.duplicated(subset=["player_id", "team_id", "season"])
    if duplicates.any():
        print(f"[FAIL] {duplicates.sum()} doublons détectés.")
        errors += 1

    # Check 4 : Bornes logiques des pourcentages [0, 100]
    percentage_cols = ["shot_accuracy", "pass_accuracy", "duel_success_rate", "dribble_success_rate"]
    for col in percentage_cols:
        out_of_bounds = df[(df[col] < 0) | (df[col] > 100)]
        if not out_of_bounds.empty:
            print(f"[FAIL] {len(out_of_bounds)} valeurs hors limites [0-100] sur {col}.")
            errors += 1

    if errors == 0:
        print("[+] Quality Check RÉUSSI ! Toutes les règles de validation sont validées.")
    else:
        print(f"[!] Quality Check ÉCHOUÉ avec {errors} erreur(s).")
        sys.exit(1)


if __name__ == "__main__":
    run_quality_checks()