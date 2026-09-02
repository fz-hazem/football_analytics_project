"""
train_umap_contextual.py
Pipeline ML Avancé :
- Normalisation par Groupe de Poste (Positional Z-Score)
- Réduction de dimension UMAP (Non-Linéaire)
"""

import os
import pandas as pd
import numpy as np
import umap
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DATABASE_URL", "sqlite:///data/football_analytics.db")
engine = create_engine(DB_URL)

def run_umap_pipeline():
    print("[*] Extraction des données statistiques...")
    
    query = """
    SELECT 
        s.player_id, p.name, p.position_group,
        s.goals_per90, s.assists_per90, s.shots_per90, s.key_passes_per90,
        s.successful_dribbles_per90, s.tackles_per90, s.interceptions_per90,
        s.blocks_per90, s.duels_won_per90, s.pass_accuracy
    FROM player_statistics s
    JOIN players p ON s.player_id = p.player_id
    WHERE s.minutes_played >= 180
    """
    
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)

    if df.empty:
        print("[!] Aucune donnée trouvée.")
        return

    feature_cols = [
        "goals_per90", "assists_per90", "shots_per90", "key_passes_per90",
        "successful_dribbles_per90", "tackles_per90", "interceptions_per90",
        "blocks_per90", "duels_won_per90", "pass_accuracy"
    ]
    
    df[feature_cols] = df[feature_cols].fillna(0)

    # 1. NORMALISATION PAR POSTE (Z-Score au sein de chaque position_group)
    print("[*] Application du Positional Z-Score...")
    df_scaled = df.copy()
    for col in feature_cols:
        df_scaled[col] = df.groupby("position_group")[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-6)
        )

    # 2. PROJECTION UMAP 2D
    print("[*] Calcul de la projection UMAP 2D...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(df_scaled[feature_cols])

    df["UMAP_1"] = embedding[:, 0]
    df["UMAP_2"] = embedding[:, 1]

    # 3. SAUVEGARDE EN BASE DE DONNÉES
    df_save = df[["player_id", "position_group", "UMAP_1", "UMAP_2"]]
    
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS player_umap_embeddings"))
        df_save.to_sql("player_umap_embeddings", conn, if_exists="replace", index=False)

    print("[+] Table 'player_umap_embeddings' générée avec succès.")

if __name__ == "__main__":
    run_umap_pipeline()