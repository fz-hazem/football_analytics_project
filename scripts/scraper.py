"""
scraper.py
Extrait les statistiques individuelles des joueurs depuis API-Football pour l'ensemble
des ligues configurées dans config/leagues.json. Sauvegarde les résultats JSON bruts dans data/raw/.
"""

import os
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_FOOTBALL_KEY")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}


def load_config_leagues():
    config_path = os.path.join("config", "leagues.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Le fichier de configuration {config_path} est introuvable.")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def scrape_league_players(league_id: int, season: int = 2023):
    print(f"[*] Scraping des statistiques pour League ID: {league_id}, Saison: {season}...")
    os.makedirs("data/raw", exist_ok=True)
    
    all_players = []
    page = 1
    total_pages = 1

    while page <= total_pages:
        url = f"{BASE_URL}/players?league={league_id}&season={season}&page={page}"
        try:
            res = requests.get(url, headers=HEADERS, timeout=15)
            
            if res.status_code == 429:
                print("[!] Rate limit atteint (HTTP 429). Pause de 10 secondes...")
                time.sleep(10)
                continue

            if res.status_code != 200:
                print(f"[!] Erreur HTTP {res.status_code} à la page {page}")
                break

            data = res.json()

            # Vérification des erreurs renvoyées dans le corps JSON par API-Football
            api_errors = data.get("errors")
            if api_errors and len(api_errors) > 0:
                print(f"[!] Erreur renvoyée par API-Football : {api_errors}")
                break

            response_data = data.get("response", [])
            
            # Limite à 3 pages max pour la version gratuite de l'API
            raw_total_pages = data.get("paging", {}).get("total", 1)
            total_pages = min(raw_total_pages, 3)

            all_players.extend(response_data)
            print(f"  -> Page {page}/{total_pages} récupérée ({len(response_data)} joueurs)")

            page += 1
            time.sleep(0.3)  # Respect des limites d'appels de l'API

        except Exception as e:
            print(f"[!] Erreur à la page {page} pour la ligue {league_id}: {e}")
            break

    output_file = f"data/raw/players_league_{league_id}_{season}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_players, f, ensure_ascii=False, indent=2)

    print(f"[+] Scraping terminé. {len(all_players)} données sauvegardées dans {output_file}\n")


def run_scraper():
    if not API_KEY or API_KEY == "votre_cle_api_ici":
        print("[!] ERREUR : La clé API_FOOTBALL_KEY n'est pas définie dans votre fichier .env")
        return

    leagues = load_config_leagues()
    for league in leagues:
        league_id = league.get("league_id")
        # Utilise la saison définie dans leagues.json si elle existe, sinon 2023 par défaut
        season = league.get("season", 2023)
        scrape_league_players(league_id=league_id, season=season)


if __name__ == "__main__":
    run_scraper()