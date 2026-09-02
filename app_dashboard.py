"""
app_dashboard.py
Tableau de bord Data Science interactif (Streamlit + Plotly)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text

st.set_page_config(page_title="Football Analytics ML Dashboard", layout="wide")

DB_URL = "sqlite:///data/football_analytics.db"
engine = create_engine(DB_URL)


@st.cache_data
def load_data():
    query = """
    SELECT 
        p.player_id, p.name, p.position_group, p.age, p.photo_url,
        c.cluster_id, c.PCA1, c.PCA2,
        o.anomaly_score,
        s.goals_per90, s.assists_per90, s.shots_per90, s.key_passes_per90,
        s.successful_dribbles_per90, s.tackles_per90, s.interceptions_per90
    FROM players p
    LEFT JOIN player_clusters c ON p.player_id = c.player_id
    LEFT JOIN player_outliers o ON p.player_id = o.player_id
    LEFT JOIN player_statistics s ON p.player_id = s.player_id
    """
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)


df = load_data()

st.title("⚽ Football Scouting & Machine Learning Platform")
st.markdown("Explorez les archétypes tactiques (K-Means/PCA), les pépites (Isolation Forest) et la similarité.")

tab1, tab2 = st.tabs(["📊 Cartographie des Clusters (PCA)", "🔍 Fiche Joueur & Similarité"])

with tab1:
    st.subheader("Projection 2D des Archétypes Tactiques")
    if not df.empty and "PCA1" in df.columns:
        fig_pca = px.scatter(
            df,
            x="PCA1",
            y="PCA2",
            color="cluster_id",
            hover_name="name",
            hover_data=["position_group", "anomaly_score"],
            color_continuous_scale="Viridis",
            title="Espace Latent des Joueurs (PCA)",
        )
        st.plotly_chart(fig_pca, use_container_width=True)

with tab2:
    player_names = df["name"].dropna().unique()
    if len(player_names) > 0:
        selected_player = st.selectbox("Sélectionnez un joueur :", player_names)
        player_data = df[df["name"] == selected_player].iloc[0]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"Profil : {player_data['name']}")
            st.write(f"**Poste :** {player_data['position_group']}")
            st.write(f"**Cluster Tactique :** {player_data['cluster_id']}")
            anomaly_val = player_data.get("anomaly_score")
            st.write(f"**Score d'Anomalie (Outlier) :** {float(anomaly_val):.4f}" if pd.notnull(anomaly_val) else "N/A")

        with col2:
            categories = ['Buts/90', 'Passes D/90', 'Tirs/90', 'Passes Clés/90', 'Dribbles/90', 'Tacles/90']
            values = [
                float(player_data['goals_per90'] or 0),
                float(player_data['assists_per90'] or 0),
                float(player_data['shots_per90'] or 0),
                float(player_data['key_passes_per90'] or 0),
                float(player_data['successful_dribbles_per90'] or 0),
                float(player_data['tackles_per90'] or 0),
            ]

            fig_radar = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself'))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False, title="Radar de Performance")
            st.plotly_chart(fig_radar, use_container_width=True)

        st.subheader("Joueurs les plus similaires (Même cluster tactique)")
        sim_query = f"""
        SELECT p.name, p.position_group, s.similarity_score, s.rank
        FROM player_similarities s
        JOIN players p ON s.similar_player_id = p.player_id
        WHERE s.player_id = '{player_data['player_id']}'
        ORDER BY s.rank ASC
        """
        with engine.connect() as conn:
            sim_df = pd.read_sql(text(sim_query), conn)

        st.dataframe(sim_df, use_container_width=True)