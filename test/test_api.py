import os
import requests
from dotenv import load_dotenv

load_dotenv()

def test_api_connection():
    api_key = os.getenv("API_FOOTBALL_KEY")
    assert api_key is not None, "La clé API_FOOTBALL_KEY est introuvable."
    
    headers = {"x-apisports-key": api_key}
    res = requests.get("https://v3.football.api-sports.io/status", headers=headers, timeout=10)
    
    assert res.status_code == 200, "Problème d'accès à l'API-Football."
    data = res.json()
    assert "response" in data