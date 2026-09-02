"""
run_pipeline.py
Orchestrateur global : Ingestion -> Clustering -> Outliers -> Similarités.
"""

import time
import subprocess

def run_script(script_name):
    print(f"\n==================================================")
    print(f"Exécution : {script_name}")
    print(f"==================================================")
    result = subprocess.run(["python", f"scripts/{script_name}"], check=True)
    return result

if __name__ == "__main__":
    start_time = time.time()

    scripts_sequence = [
        "quality_check.py",
        "load_to_postgres.py",
        "train_player_clustering.py",
        "detect_scouting_outliers.py",
        "build_player_similarities.py"
    ]

    for script in scripts_sequence:
        run_script(script)

    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Pipeline ML complet exécuté avec succès en {elapsed:.2f} secondes.")