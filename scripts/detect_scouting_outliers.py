"""
detect_scouting_outliers.py
Détection d'anomalies (Outliers) avec Isolation Forest.
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DATABASE_URL", "sqlite:///data/football_analytics.db")
engine = create_engine(DB_URL)


def detect_outliers(top_n: int = 10):
    print("[*] Chargement des données statistiques et des clusters...")

    query = """
    SELECT 
        p.player_id, p.name, p.position_group, c.cluster_id,
        s.goals_per90, s.assists_per90, s.shots_per90, s.shots_on_target_per90,
        s.goal_contributions_per90, s.key_passes_per90, s.successful_dribbles_per90,
        s.tackles_per90, s.interceptions_per90, s.blocks_per90, s.defensive_actions_per90,
        s.duels_won_per90, s.fouls_drawn_per90, s.fouls_committed_per90,
        s.shot_accuracy, s.goal_conversion, s.pass_accuracy, s.dribble_success_rate
    FROM player_statistics s
    JOIN players p ON s.player_id = p.player_id
    LEFT JOIN player_clusters c ON s.player_id = c.player_id
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
        "duels_won_per90", "fouls_drawn_per90", "fouls_committed_per90",
        "shot_accuracy", "goal_conversion", "pass_accuracy", "dribble_success_rate"
    ]

    df[feature_cols] = df[feature_cols].fillna(0)
    df_grouped = df.groupby(["player_id", "name", "position_group", "cluster_id"])[feature_cols].mean().reset_index()

    # Révélation du joueur isolé dans le Cluster 4
    cluster_4_player = df_grouped[df_grouped["cluster_id"] == 4]
    if not cluster_4_player.empty:
        print("\n🔍 JOUEUR UNIQUE DU CLUSTER 4 :")
        for _, row in cluster_4_player.iterrows():
            print(f" -> {row['name']} ({row['position_group']})")
        print("-" * 50)

    # Entraînement Isolation Forest
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_grouped[feature_cols])

    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    df_grouped["outlier_score"] = iso_forest.fit_predict(scaled_features)
    df_grouped["anomaly_score"] = iso_forest.decision_function(scaled_features)

    outliers = df_grouped.sort_values(by="anomaly_score", ascending=True).head(top_n)

    print(f"\n🔥 TOP {top_n} DES PROFILS LES PLUS ATYPIQUES (SCOUTING OUTLIERS) :")
    for rank, (_, row) in enumerate(outliers.iterrows(), start=1):
        print(f"{rank:2d}. {row['name']:<25} | Cluster: {row['cluster_id']} | Score: {row['anomaly_score']:.4f}")

    # Sauvegarde SQLite
    df_save = df_grouped[["player_id", "anomaly_score", "outlier_score"]]
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS player_outliers"))
        df_save.to_sql("player_outliers", conn, if_exists="replace", index=False)

    print("\n[+] Table 'player_outliers' mise à jour avec succès.")


if __name__ == "__main__":
    detect_outliers()