"""
train_player_clustering.py
Réduction de dimension (PCA) et Clustering (K-Means) pour identifier 
les archétypes tactiques des joueurs et sauvegarder la carte d'analyse.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DATABASE_URL", "sqlite:///data/football_analytics.db")
engine = create_engine(DB_URL)


def run_tactical_clustering(n_clusters: int = 5):
    print("[*] Extraction des métriques statistiques depuis SQLite...")

    query = """
    SELECT 
        s.player_id, p.name, p.position_group,
        s.goals_per90, s.assists_per90, s.shots_per90, s.shots_on_target_per90,
        s.goal_contributions_per90, s.key_passes_per90, s.successful_dribbles_per90,
        s.tackles_per90, s.interceptions_per90, s.blocks_per90, s.defensive_actions_per90,
        s.duels_won_per90, s.fouls_drawn_per90, s.fouls_committed_per90, s.cards_per90,
        s.shot_accuracy, s.goal_conversion, s.key_pass_rate, s.pass_accuracy,
        s.duel_success_rate, s.dribble_success_rate
    FROM player_statistics s
    JOIN players p ON s.player_id = p.player_id
    WHERE s.minutes_played >= 180
    """

    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)

    if df.empty:
        print("[!] Aucune donnée statistique disponible.")
        return

    feature_cols = [
        "goals_per90", "assists_per90", "shots_per90", "shots_on_target_per90",
        "goal_contributions_per90", "key_passes_per90", "successful_dribbles_per90",
        "tackles_per90", "interceptions_per90", "blocks_per90", "defensive_actions_per90",
        "duels_won_per90", "fouls_drawn_per90", "fouls_committed_per90", "cards_per90",
        "shot_accuracy", "goal_conversion", "key_pass_rate", "pass_accuracy",
        "duel_success_rate", "dribble_success_rate"
    ]

    # Remplacement des valeurs nuls et agrégation par joueur unique
    df[feature_cols] = df[feature_cols].fillna(0)
    df_grouped = df.groupby(["player_id", "name", "position_group"])[feature_cols].mean().reset_index()

    # 1. Normalisation
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_grouped[feature_cols])

    # 2. Réduction de Dimension avec PCA (2 composants pour affichage graphique)
    pca = PCA(n_components=2, random_state=42)
    pca_features = pca.fit_transform(scaled_data)
    df_grouped["PCA1"] = pca_features[:, 0]
    df_grouped["PCA2"] = pca_features[:, 1]

    variance_explained = pca.explained_variance_ratio_ * 100
    print(f"[+] PCA calculé : {variance_explained[0]:.1f}% (Axe 1) et {variance_explained[1]:.1f}% (Axe 2) de variance expliquée.")

    # 3. Clustering K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_grouped["cluster_id"] = kmeans.fit_predict(scaled_data)

    # 4. Identification des caractéristiques dominantes par cluster
    print("\n--- ANALYSE DES ARCHÉTYPES TACTIQUES TROUVÉS ---")
    cluster_means = df_grouped.groupby("cluster_id")[feature_cols].mean()
    
    for cluster_num in range(n_clusters):
        top_features = cluster_means.loc[cluster_num].nlargest(3).index.tolist()
        player_count = (df_grouped['cluster_id'] == cluster_num).sum()
        print(f"Cluster {cluster_num} ({player_count} joueurs) -> Profil dominé par : {', '.join(top_features)}")

    # 5. Sauvegarde en BDD SQLite
    df_db = df_grouped[["player_id", "cluster_id", "PCA1", "PCA2"]]
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS player_clusters"))
        df_db.to_sql("player_clusters", conn, if_exists="replace", index=False)
    print(f"\n[+] Résultats de clustering enregistrés dans la table 'player_clusters'.")

    # 6. Génération et Sauvegarde du Graphique Scatter Plot PCA
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="darkgrid")
    
    palette = sns.color_palette("viridis", n_colors=n_clusters)
    scatter = sns.scatterplot(
        data=df_grouped,
        x="PCA1",
        y="PCA2",
        hue="cluster_id",
        palette=palette,
        style="position_group",
        s=100,
        alpha=0.85
    )

    # Annotation de quelques joueurs repères
    top_players = df_grouped.head(10)
    for _, row in top_players.iterrows():
        plt.text(row["PCA1"] + 0.1, row["PCA2"] + 0.1, row["name"], fontsize=8, color="black", weight="bold")

    plt.title("Cartographie des Archétypes Tactiques des Joueurs (PCA + K-Means)", fontsize=14, weight="bold")
    plt.xlabel(f"Composante Principale 1 ({variance_explained[0]:.1f}% variance)")
    plt.ylabel(f"Composante Principale 2 ({variance_explained[1]:.1f}% variance)")
    plt.legend(title="Cluster / Poste", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    output_path = "data/player_archetypes_pca.png"
    plt.savefig(output_path, dpi=300)
    print(f"[+] Graphique sauvegardé avec succès dans '{output_path}'.")


if __name__ == "__main__":
    run_tactical_clustering(n_clusters=5)