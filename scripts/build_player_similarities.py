"""
build_player_similarities.py
Calcule la similarité cosinus inter-joueurs restreinte par Cluster Tactique.
"""

import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DATABASE_URL", "sqlite:///data/football_analytics.db")
engine = create_engine(DB_URL)


def calculate_cluster_aware_similarities(top_k: int = 5):
    print("[*] Calcul des similarités intra-clusters (Cosine Similarity)...")

    query = """
    SELECT 
        s.player_id, p.name, c.cluster_id,
        s.goals_per90, s.assists_per90, s.shots_per90, s.shots_on_target_per90,
        s.goal_contributions_per90, s.key_passes_per90, s.successful_dribbles_per90,
        s.tackles_per90, s.interceptions_per90, s.blocks_per90, s.defensive_actions_per90,
        s.duels_won_per90, s.fouls_drawn_per90, s.fouls_committed_per90,
        s.shot_accuracy, s.goal_conversion, s.pass_accuracy, s.dribble_success_rate
    FROM player_statistics s
    JOIN players p ON s.player_id = p.player_id
    JOIN player_clusters c ON s.player_id = c.player_id
    WHERE s.minutes_played >= 180
    """

    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)

    if df.empty:
        print("[!] Aucune donnée disponible. Lancez d'abord train_player_clustering.py")
        return

    feature_cols = [
        "goals_per90", "assists_per90", "shots_per90", "shots_on_target_per90",
        "goal_contributions_per90", "key_passes_per90", "successful_dribbles_per90",
        "tackles_per90", "interceptions_per90", "blocks_per90", "defensive_actions_per90",
        "duels_won_per90", "fouls_drawn_per90", "fouls_committed_per90",
        "shot_accuracy", "goal_conversion", "pass_accuracy", "dribble_success_rate"
    ]

    df[feature_cols] = df[feature_cols].fillna(0)
    df_grouped = df.groupby(["player_id", "name", "cluster_id"])[feature_cols].mean().reset_index()

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_grouped[feature_cols])

    similarities = []

    # Calcul intra-cluster
    for cluster_id in df_grouped["cluster_id"].unique():
        cluster_indices = df_grouped[df_grouped["cluster_id"] == cluster_id].index.tolist()
        
        if len(cluster_indices) <= 1:
            continue

        sub_matrix = scaled_data[cluster_indices]
        sim_matrix = cosine_similarity(sub_matrix)

        for i_loc, idx_a in enumerate(cluster_indices):
            player_a_id = df_grouped.loc[idx_a, "player_id"]
            
            # Trier par similarité décroissante
            scores = sim_matrix[i_loc]
            sorted_indices = np.argsort(scores)[::-1]

            rank = 1
            for j_loc in sorted_indices:
                idx_b = cluster_indices[j_loc]
                if idx_a == idx_b:
                    continue

                player_b_id = df_grouped.loc[idx_b, "player_id"]
                similarity_score = float(scores[j_loc])

                similarities.append({
                    "player_id": player_a_id,
                    "similar_player_id": player_b_id,
                    "similarity_score": round(similarity_score, 4),
                    "rank": rank
                })

                rank += 1
                if rank > top_k:
                    break

    sim_df = pd.DataFrame(similarities)

    # Sauvegarde dans la table SQLite
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS player_similarities"))
        sim_df.to_sql("player_similarities", conn, if_exists="replace", index=False)

    print(f"[+] {len(sim_df)} relations de similarité intra-cluster insérées dans 'player_similarities'.")


if __name__ == "__main__":
    calculate_cluster_aware_similarities()