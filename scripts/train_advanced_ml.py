"""
train_advanced_ml.py
Pipeline ML Avancé : 
- QuantileTransformer (Percentile Ranks)
- Gaussian Mixture Models (Soft Clustering Probabiliste)
- Similarité Hybride & Explicabilité des traits tactiques
"""

import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.preprocessing import QuantileTransformer
from sklearn.mixture import GaussianMixture
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DATABASE_URL", "sqlite:///data/football_analytics.db")
engine = create_engine(DB_URL)


def run_advanced_ml_pipeline(n_components: int = 5):
    print("[*] Extraction des métriques pour le pipeline ML Avancé...")

    query = """
    SELECT 
        s.player_id, p.name, p.position_group,
        s.goals_per90, s.assists_per90, s.shots_per90, s.shots_on_target_per90,
        s.goal_contributions_per90, s.key_passes_per90, s.successful_dribbles_per90,
        s.tackles_per90, s.interceptions_per90, s.blocks_per90, s.defensive_actions_per90,
        s.duels_won_per90, s.fouls_drawn_per90, s.shot_accuracy, s.pass_accuracy,
        s.duel_success_rate, s.dribble_success_rate
    FROM player_statistics s
    JOIN players p ON s.player_id = p.player_id
    WHERE s.minutes_played >= 180
    """

    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)

    if df.empty:
        print("[!] Aucune donnée disponible.")
        return

    feature_cols = [
        "goals_per90", "assists_per90", "shots_per90", "shots_on_target_per90",
        "goal_contributions_per90", "key_passes_per90", "successful_dribbles_per90",
        "tackles_per90", "interceptions_per90", "blocks_per90", "defensive_actions_per90",
        "duels_won_per90", "fouls_drawn_per90", "shot_accuracy", "pass_accuracy",
        "duel_success_rate", "dribble_success_rate"
    ]

    df[feature_cols] = df[feature_cols].fillna(0)
    df_grouped = df.groupby(["player_id", "name", "position_group"])[feature_cols].mean().reset_index()

    # 1. TRANSFORMATION PAR CENTILES
    print("[*] Application du QuantileTransformer (Percentile Ranking)...")
    quantile_scaler = QuantileTransformer(n_quantiles=len(df_grouped), output_distribution='uniform', random_state=42)
    percentile_data = quantile_scaler.fit_transform(df_grouped[feature_cols])

    # 2. SOFT CLUSTERING (GMM) - ENTRAÎNEMENT PUIS PRÉDICTION
    print(f"[*] Entraînement du GMM (Soft Clustering sur {n_components} composants)...")
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(percentile_data)  # Entraînement préalable
    probabilities = gmm.predict_proba(percentile_data)

    df_grouped["primary_cluster"] = np.argmax(probabilities, axis=1)
    df_grouped["cluster_confidence"] = np.max(probabilities, axis=1)

    prob_strings = []
    for row in probabilities:
        prob_dict = {f"cluster_{i}": round(float(p), 3) for i, p in enumerate(row)}
        prob_strings.append(str(prob_dict))
    df_grouped["all_cluster_probabilities"] = prob_strings

    print("\n--- ÉCHANTILLON DU SOFT CLUSTERING ---")
    for _, row in df_grouped.head(5).iterrows():
        print(f"Joueur: {row['name']:<20} | Cluster Dominant: {row['primary_cluster']} (Confiance: {row['cluster_confidence']*100:.1f}%)")

    # 3. SAUVEGARDE DANS SQLITE
    df_save_gmm = df_grouped[["player_id", "primary_cluster", "cluster_confidence", "all_cluster_probabilities"]]
    
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS player_gmm_clusters"))
        df_save_gmm.to_sql("player_gmm_clusters", conn, if_exists="replace", index=False)

    print("\n[+] Table 'player_gmm_clusters' générée avec succès.")


if __name__ == "__main__":
    run_advanced_ml_pipeline()