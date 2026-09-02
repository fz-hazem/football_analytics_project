"""
load_to_postgres.py
Crée la structure relationnelle SQLite / PostgreSQL et charge les tables dimensionnelles et de faits.
"""

import os
import json
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

# Si DATABASE_URL n'est pas défini, on utilise SQLite par défaut
DB_URL = os.getenv("DATABASE_URL", "sqlite:///data/football_analytics.db")

# Assurer que le dossier 'data' existe pour SQLite
os.makedirs("data", exist_ok=True)

engine = create_engine(DB_URL)


def init_db_schema():
    print(f"[*] Initialisation du schéma sur la base : {DB_URL}")
    
    # Adaptation du type auto-incrémenté pour SQLite / Postgres
    is_sqlite = DB_URL.startswith("sqlite")
    serial_type = "INTEGER PRIMARY KEY AUTOINCREMENT" if is_sqlite else "SERIAL PRIMARY KEY"

    schema_sql = f"""
    CREATE TABLE IF NOT EXISTS leagues (
        league_id INT PRIMARY KEY,
        name VARCHAR(100),
        country VARCHAR(100),
        category VARCHAR(50)
    );

    CREATE TABLE IF NOT EXISTS teams (
        team_id INT PRIMARY KEY,
        name VARCHAR(100),
        logo_url VARCHAR(255)
    );

    CREATE TABLE IF NOT EXISTS players (
        player_id INT PRIMARY KEY,
        name VARCHAR(100),
        age INT,
        position_group VARCHAR(20),
        detailed_position VARCHAR(50),
        photo_url VARCHAR(255)
    );

    CREATE TABLE IF NOT EXISTS player_statistics (
        stat_id {serial_type},
        player_id INT REFERENCES players(player_id),
        team_id INT REFERENCES teams(team_id),
        league_id INT REFERENCES leagues(league_id),
        season INT,
        appearances INT,
        starts INT,
        minutes_played INT,
        rating FLOAT,
        
        goals INT, assists INT, shots INT, shots_on_target INT,
        key_passes INT, passes_total INT, dribbles_succ INT,
        tackles INT, interceptions INT, blocks INT,
        duels_won INT, fouls_drawn INT, fouls_committed INT, cards INT,
        
        goals_per90 FLOAT, assists_per90 FLOAT, shots_per90 FLOAT,
        shots_on_target_per90 FLOAT, goal_contributions_per90 FLOAT,
        key_passes_per90 FLOAT, successful_dribbles_per90 FLOAT,
        tackles_per90 FLOAT, interceptions_per90 FLOAT, blocks_per90 FLOAT,
        defensive_actions_per90 FLOAT, duels_won_per90 FLOAT,
        fouls_drawn_per90 FLOAT, fouls_committed_per90 FLOAT, cards_per90 FLOAT,
        
        shot_accuracy FLOAT, goal_conversion FLOAT, key_pass_rate FLOAT,
        pass_accuracy FLOAT, duel_success_rate FLOAT, dribble_success_rate FLOAT,

        CONSTRAINT unique_player_team_season UNIQUE (player_id, team_id, season)
    );

    CREATE TABLE IF NOT EXISTS player_similarities (
        player_id INT REFERENCES players(player_id),
        similar_player_id INT REFERENCES players(player_id),
        rank_order INT,
        similarity_score FLOAT,
        PRIMARY KEY (player_id, rank_order)
    );
    """
    with engine.begin() as conn:
        # Exécuter chaque instruction SQL séparément pour éviter les incompatibilités SQLite
        for statement in schema_sql.split(";"):
            if statement.strip():
                conn.execute(text(statement.strip()))
    print("[+] Schéma de base de données configuré avec succès.")


def insert_new_records(df: pd.DataFrame, table_name: str, pks: list, conn):
    """Insère uniquement les nouvelles lignes qui n'existent pas encore dans la table."""
    if df.empty:
        return

    pk_cols = ", ".join(pks)
    try:
        existing = pd.read_sql(f"SELECT {pk_cols} FROM {table_name}", conn)
    except Exception:
        existing = pd.DataFrame()

    if not existing.empty:
        cond = df.set_index(pks).index.isin(existing.set_index(pks).index)
        df_to_insert = df[~cond]
    else:
        df_to_insert = df

    if not df_to_insert.empty:
        df_to_insert.to_sql(table_name, conn, if_exists="append", index=False)
        print(f"  [+] {len(df_to_insert)} nouvelles lignes insérées dans '{table_name}'.")
    else:
        print(f"  [-] Aucune nouvelle donnée à insérer dans '{table_name}'.")


def load_data():
    print("[*] Chargement des données dans la base de données...")

    processed_file = "data/processed/players_processed.csv"
    if not os.path.exists(processed_file):
        raise FileNotFoundError(f"Le fichier {processed_file} est introuvable. Exécutez transform.py d'abord.")

    # 1. Ligues
    with open("config/leagues.json", "r", encoding="utf-8") as f:
        leagues_data = json.load(f)
    df_leagues = pd.DataFrame(leagues_data)

    for col in ["country", "category"]:
        if col not in df_leagues.columns:
            df_leagues[col] = "Non renseigné"

    df_leagues = df_leagues[["league_id", "name", "country", "category"]].drop_duplicates("league_id")

    # 2. Joueurs & Stats
    df_processed = pd.read_csv(processed_file)

    # Extraire la dimension Teams
    df_teams = df_processed[["team_id", "team_name", "team_logo"]].drop_duplicates("team_id").copy()
    df_teams.columns = ["team_id", "name", "logo_url"]

    # Extraire la dimension Players
    df_players = df_processed[["player_id", "name", "age", "position_group", "detailed_position", "photo_url"]].drop_duplicates("player_id").copy()

    # Colonnes de statistiques
    stat_cols = [
        "player_id", "team_id", "league_id", "season", "appearances", "starts",
        "minutes_played", "rating", "goals", "assists", "shots", "shots_on_target",
        "key_passes", "passes_total", "dribbles_succ", "tackles", "interceptions",
        "blocks", "duels_won", "fouls_drawn", "fouls_committed", "cards",
        "goals_per90", "assists_per90", "shots_per90", "shots_on_target_per90",
        "goal_contributions_per90", "key_passes_per90", "successful_dribbles_per90",
        "tackles_per90", "interceptions_per90", "blocks_per90", "defensive_actions_per90",
        "duels_won_per90", "fouls_drawn_per90", "fouls_committed_per90", "cards_per90",
        "shot_accuracy", "goal_conversion", "key_pass_rate", "pass_accuracy",
        "duel_success_rate", "dribble_success_rate"
    ]
    df_stats = df_processed[stat_cols].drop_duplicates(subset=["player_id", "team_id", "season"]).copy()

    # Ingestion dans la base
    with engine.begin() as conn:
        insert_new_records(df_leagues, "leagues", ["league_id"], conn)
        insert_new_records(df_teams, "teams", ["team_id"], conn)
        insert_new_records(df_players, "players", ["player_id"], conn)
        insert_new_records(df_stats, "player_statistics", ["player_id", "team_id", "season"], conn)

    print("[+] Chargement terminé avec succès.")


if __name__ == "__main__":
    init_db_schema()
    load_data()