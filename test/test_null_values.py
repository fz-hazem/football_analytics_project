import os
import pandas as pd

def test_processed_null_thresholds():
    file_path = "data/processed/players_processed.csv"
    assert os.path.exists(file_path), "Fichier players_processed.csv introuvable."
    
    df = pd.read_csv(file_path)
    
    # Les identifiants critiques ne doivent contenir aucune valeur nulle
    critical_cols = ["player_id", "team_id", "league_id", "minutes_played", "position_group"]
    for col in critical_cols:
        assert df[col].isnull().sum() == 0, f"Présence de nulls interdits dans la colonne {col}."