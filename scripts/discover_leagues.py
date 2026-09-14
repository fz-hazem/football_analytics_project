"""
discover_leagues.py
Explore les compétitions disponibles sur API-Football et génère un bilan des ligues actives.
"""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_FOOTBALL_KEY")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}


def discover_available_leagues(season: int = 2026):
    print(f"[*] Recherche des ligues disponibles pour la saison {season}...")
    url = f"{BASE_URL}/leagues?season={season}"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json().get("response", [])
        
        leagues_summary = []
        for item in data:
            league = item["league"]
            country = item["country"]
            leagues_summary.append({
                "league_id": league["id"],
                "name": league["name"],
                "type": league["type"],
                "country": country["name"]
            })
            
        print(f"[+] {len(leagues_summary)} ligues identifiées.")
        return leagues_summary
    except Exception as e:
        print(f"[!] Erreur lors de la récupération des ligues: {e}")
        return []


if __name__ == "__main__":
    discover_available_leagues(2026)