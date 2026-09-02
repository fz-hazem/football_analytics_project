"""
transform.py
Transforme les JSONs bruts en DataFrame structuré.
Calcule :
- 15 Métriques par 90 minutes (Per-90)
- 6 Métriques d'Efficacité (Shot accuracy, Pass accuracy, etc.)
- Normalisation du positionnement (Attacker, Midfielder, Defender)
"""

import os
import glob
import json
import pandas as pd
import numpy as np


def map_position_group(position: str) -> str:
    if not position:
        return "Midfielder"
    pos = str(position).lower()
    if any(k in pos for k in ["attacker", "forward", "striker", "wing"]):
        return "Attacker"
    elif any(k in pos for k in ["defender", "back"]):
        return "Defender"
    return "Midfielder"


def safe_float(value, default: float = 0.0) -> float:
    """Convertit de manière sécurisée une valeur (chaîne, int, float) en float."""
    if value is None:
        return default
    try:
        clean_val = str(value).replace("%", "").strip()
        return float(clean_val)
    except (ValueError, TypeError):
        return default


def process_raw_data() -> pd.DataFrame:
    print("[*] Début de la transformation et ingénierie des métriques...")
    json_files = glob.glob("data/raw/players_league_*.json")
    
    if not json_files:
        raise FileNotFoundError("Aucun fichier JSON brut trouvé dans data/raw/")

    records = []
    
    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for entry in data:
            player = entry.get("player", {})
            stats_list = entry.get("statistics", [])
            if not stats_list:
                continue

            stats = stats_list[0]
            minutes = stats.get("games", {}).get("minutes") or 0
            
            # Filtre : Minimum 90 minutes jouées
            if minutes < 90:
                continue

            p90_factor = 90.0 / minutes

            # Extraction des événements bruts
            goals = stats.get("goals", {}).get("total") or 0
            assists = stats.get("goals", {}).get("assists") or 0
            shots = stats.get("shots", {}).get("total") or 0
            shots_on_target = stats.get("shots", {}).get("on") or 0
            
            passes_total = stats.get("passes", {}).get("total") or 0
            passes_acc_raw = stats.get("passes", {}).get("accuracy")
            passes_acc = safe_float(passes_acc_raw, 0.0)
            key_passes = stats.get("passes", {}).get("key") or 0
            
            dribbles_att = stats.get("dribbles", {}).get("attempts") or 0
            dribbles_succ = stats.get("dribbles", {}).get("success") or 0
            
            tackles = stats.get("tackles", {}).get("total") or 0
            interceptions = stats.get("tackles", {}).get("interceptions") or 0
            blocks = stats.get("tackles", {}).get("blocks") or 0
            
            duels_tot = stats.get("duels", {}).get("total") or 0
            duels_won = stats.get("duels", {}).get("won") or 0
            
            fouls_comm = stats.get("fouls", {}).get("committed") or 0
            fouls_drawn = stats.get("fouls", {}).get("drawn") or 0
            cards_yellow = stats.get("cards", {}).get("yellow") or 0
            cards_red = stats.get("cards", {}).get("red") or 0
            total_cards = cards_yellow + cards_red

            raw_pos = stats.get("games", {}).get("position") or "Midfielder"

            rec = {
                # Identité & Contexte
                "player_id": player.get("id"),
                "name": player.get("name", "Unknown"),
                "age": player.get("age"),
                "photo_url": player.get("photo"),
                "position_group": map_position_group(raw_pos),
                "detailed_position": raw_pos,
                "team_id": stats.get("team", {}).get("id"),
                "team_name": stats.get("team", {}).get("name"),
                "team_logo": stats.get("team", {}).get("logo"),
                "league_id": stats.get("league", {}).get("id"),
                "season": stats.get("league", {}).get("season", 2023),
                "appearances": stats.get("games", {}).get("appearences") or 0,
                "starts": stats.get("games", {}).get("lineups") or 0,
                "minutes_played": minutes,
                "rating": safe_float(stats.get("games", {}).get("rating"), None),

                # Statistiques brutes
                "goals": goals,
                "assists": assists,
                "shots": shots,
                "shots_on_target": shots_on_target,
                "key_passes": key_passes,
                "passes_total": passes_total,
                "dribbles_succ": dribbles_succ,
                "tackles": tackles,
                "interceptions": interceptions,
                "blocks": blocks,
                "duels_won": duels_won,
                "fouls_drawn": fouls_drawn,
                "fouls_committed": fouls_comm,
                "cards": total_cards,

                # 15 Métriques Per-90
                "goals_per90": round(goals * p90_factor, 2),
                "assists_per90": round(assists * p90_factor, 2),
                "shots_per90": round(shots * p90_factor, 2),
                "shots_on_target_per90": round(shots_on_target * p90_factor, 2),
                "goal_contributions_per90": round((goals + assists) * p90_factor, 2),
                "key_passes_per90": round(key_passes * p90_factor, 2),
                "successful_dribbles_per90": round(dribbles_succ * p90_factor, 2),
                "tackles_per90": round(tackles * p90_factor, 2),
                "interceptions_per90": round(interceptions * p90_factor, 2),
                "blocks_per90": round(blocks * p90_factor, 2),
                "defensive_actions_per90": round((tackles + interceptions + blocks) * p90_factor, 2),
                "duels_won_per90": round(duels_won * p90_factor, 2),
                "fouls_drawn_per90": round(fouls_drawn * p90_factor, 2),
                "fouls_committed_per90": round(fouls_comm * p90_factor, 2),
                "cards_per90": round(total_cards * p90_factor, 2),

                # 6 Métriques d'Efficacité (%)
                "shot_accuracy": round((shots_on_target / shots * 100), 2) if shots > 0 else 0.0,
                "goal_conversion": round((goals / shots * 100), 2) if shots > 0 else 0.0,
                "key_pass_rate": round((key_passes / passes_total * 100), 2) if passes_total > 0 else 0.0,
                "pass_accuracy": passes_acc,
                "duel_success_rate": round((duels_won / duels_tot * 100), 2) if duels_tot > 0 else 0.0,
                "dribble_success_rate": round((dribbles_succ / dribbles_att * 100), 2) if dribbles_att > 0 else 0.0,
            }
            records.append(rec)

    df = pd.DataFrame(records)
    
    if df.empty:
        print("[!] AVERTISSEMENT : Aucun enregistrement n'a été conservé après filtrage.")
        return df

    # Suppression des doublons potentiels (Même joueur, même équipe, même saison)
    df.drop_duplicates(subset=["player_id", "team_id", "season"], inplace=True)
    
    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/players_processed.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[+] Transformation réussie : {len(df)} enregistrements sauvegardés dans {output_path}")
    return df


if __name__ == "__main__":
    process_raw_data()