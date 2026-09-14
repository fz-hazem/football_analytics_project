import requests
import pandas as pd
import urllib.parse
from sqlalchemy import create_engine

# ==========================================
# 1. CONFIGURATION
# ==========================================
API_KEY = "2e4b316379b42a420f84aefa76216bec"
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

# Configuration PostgreSQL
DB_USER = "postgres"
DB_PASSWORD = "VOTRE_MOT_DE_PASSE"  # <-- Remplacez par votre vrai mot de passe PostgreSQL
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "football_analytics"       # <-- Nom exact de votre base dans pgAdmin

# L'encodage via quote_plus() corrige l'erreur UnicodeDecodeError ('é') sous Windows
encoded_password = urllib.parse.quote_plus(DB_PASSWORD)
DB_URI = f"postgresql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DB_URI)


# ==========================================
# 2. EXTRACTION (API REST)
# ==========================================
def fetch_league_players(league_id: int, season: int, page: int = 1):
    """Récupère une page de données depuis API-Football."""
    url = f"{BASE_URL}/players?league={league_id}&season={season}&page={page}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Erreur API ({response.status_code}) à la page {page}")
            return None
    except Exception as e:
        print(f"Erreur de connexion : {e}")
        return None


# ==========================================
# 3. TRANSFORMATION & CALCULS (/90 ET %)
# ==========================================
def process_api_response(raw_json: dict):
    """Extrait les tables de dimensions et calcule les métriques normalisées par 90 min."""
    if not raw_json or "response" not in raw_json:
        return [], [], [], []

    leagues, teams, players, stats = [], [], [], []

    for item in raw_json["response"]:
        p = item["player"]
        
        if not item.get("statistics"):
            continue
            
        s = item["statistics"][0]
        minutes = s["games"]["minutes"] or 0

        # Filtre sur le temps de jeu (minimum 180 min pour la pertinence)
        if minutes < 180:
            continue

        # --- Dimensions ---
        players.append({
            "player_id": p["id"],
            "name": p["name"],
            "age": p["age"],
            "nationality": p["nationality"],
            "position": s["games"]["position"]
        })

        teams.append({
            "team_id": s["team"]["id"],
            "name": s["team"]["name"],
            "country": None,
            "logo_url": s["team"]["logo"]
        })

        leagues.append({
            "league_id": s["league"]["id"],
            "name": s["league"]["name"],
            "country": s["league"]["country"],
            "season": s["league"]["season"]
        })

        # --- Métriques Calculées (/90 et %) ---
        p90_factor = 90.0 / minutes if minutes > 0 else 0
        
        goals = s["goals"]["total"] or 0
        assists = s["goals"]["assists"] or 0
        key_passes = s["passes"]["key"] or 0
        tackles = s["tackles"]["total"] or 0
        
        passes_acc = s["passes"]["accuracy"] or 0
        
        duels_tot = s["duels"]["total"] or 0
        duels_won = s["duels"]["won"] or 0
        duel_pct = round((duels_won / duels_tot * 100), 2) if duels_tot > 0 else 0.0

        stats.append({
            "player_id": p["id"],
            "team_id": s["team"]["id"],
            "league_id": s["league"]["id"],
            "season": s["league"]["season"],
            "minutes_played": minutes,
            "goals_total": goals,
            "assists_total": assists,
            "goals_per_90": round(goals * p90_factor, 2),
            "assists_per_90": round(assists * p90_factor, 2),
            "key_passes_per_90": round(key_passes * p90_factor, 2),
            "tackles_per_90": round(tackles * p90_factor, 2),
            "pass_accuracy_pct": float(passes_acc),
            "duel_win_pct": duel_pct
        })

    return leagues, teams, players, stats


# ==========================================
# 4. CHARGEMENT POSTGRESQL (LOAD)
# ==========================================
def load_to_postgres(leagues, teams, players, stats):
    """Insère les données dans la BDD en évitant les conflits de clés primaires."""
    df_leagues = pd.DataFrame(leagues).drop_duplicates(subset=["league_id"]) if leagues else pd.DataFrame()
    df_teams = pd.DataFrame(teams).drop_duplicates(subset=["team_id"]) if teams else pd.DataFrame()
    df_players = pd.DataFrame(players).drop_duplicates(subset=["player_id"]) if players else pd.DataFrame()
    df_stats = pd.DataFrame(stats).drop_duplicates(subset=["player_id", "team_id", "league_id", "season"]) if stats else pd.DataFrame()

    if df_stats.empty:
        print("Aucune donnée à charger.")
        return

    try:
        with engine.begin() as conn:
            # Vérification des IDs déjà existants
            existing_leagues = pd.read_sql("SELECT league_id FROM dim_leagues", conn)["league_id"].tolist()
            existing_teams = pd.read_sql("SELECT team_id FROM dim_teams", conn)["team_id"].tolist()
            existing_players = pd.read_sql("SELECT player_id FROM dim_players", conn)["player_id"].tolist()

            # Filtrage des lignes existantes
            df_leagues = df_leagues[~df_leagues["league_id"].isin(existing_leagues)]
            df_teams = df_teams[~df_teams["team_id"].isin(existing_teams)]
            df_players = df_players[~df_players["player_id"].isin(existing_players)]

            # Insertion des dimensions
            if not df_leagues.empty:
                df_leagues.to_sql("dim_leagues", conn, if_exists="append", index=False)
            if not df_teams.empty:
                df_teams.to_sql("dim_teams", conn, if_exists="append", index=False)
            if not df_players.empty:
                df_players.to_sql("dim_players", conn, if_exists="append", index=False)
            
            # Insertion des faits
            df_stats.to_sql("fact_player_stats", conn, if_exists="append", index=False)

        print(f"Chargement réussi : {len(df_players)} nouveaux joueurs et {len(df_stats)} lignes de statistiques insérés.")

    except Exception as e:
        print(f"Erreur lors de l'insertion dans la base de données : {e}")


# ==========================================
# 5. EXECUTION
# ==========================================
if __name__ == "__main__":
    LEAGUE_ID = 61  # 61 = Ligue 1, 39 = Premier League, 140 = La Liga
    SEASON = 2023   # Année de la saison

    print(f"Démarrage de l'ETL (Ligue ID: {LEAGUE_ID}, Saison: {SEASON})...")

    all_leagues, all_teams, all_players, all_stats = [], [], [], []

    # Extraction des 5 premières pages (environ 100 joueurs)
    for page in range(1, 6):
        print(f"Page {page}/5 en cours...")
        data = fetch_league_players(LEAGUE_ID, SEASON, page)
        
        if data and data.get("response"):
            lg, tm, pl, st = process_api_response(data)
            all_leagues.extend(lg)
            all_teams.extend(tm)
            all_players.extend(pl)
            all_stats.extend(st)

    print("Chargement dans PostgreSQL...")
    load_to_postgres(all_leagues, all_teams, all_players, all_stats)