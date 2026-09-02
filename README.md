<div align="center">

# ⚽ Machine Learning Football Scouting & Analytics Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14%2B-black.svg)](https://nextjs.org/)
[![Prisma](https://img.shields.io/badge/Prisma-5%2B-2D3748.svg)](https://www.prisma.io/)
[![SQLite](https://img.shields.io/badge/SQLite-3-lightgrey.svg)](https://www.sqlite.org/)

An end-to-end, professional-grade football scouting and analytics platform designed for modern, data-driven recruitment. Moving beyond simple web interfaces, this engine processes raw match statistics into high-dimensional tactical embeddings, probabilistic soft clusters, outlier gem detection, and contextual similarity models.

</div>

---

## 📖 Engine Overview & Pipeline Execution Flow

The machine learning engine operates as a sequential execution pipeline, beginning with `quality_check.py` to enforce statistical validation rules and schema compliance before `load_to_postgres.py` ingests the cleaned player data into the SQLite database. Dimensionality reduction and hard segmentation are executed by `train_player_clustering.py`, which applies PCA and K-Means to map players into five core tactical archetypes, while `detect_scouting_outliers.py` utilizes an Isolation Forest algorithm to identify statistical anomalies and hidden scouting gems. To power replacement search functionality, `build_player_similarities.py` computes intra-cluster cosine similarity scores across matched tactical profiles. For probabilistic modeling, `train_advanced_ml.py` converts skewed per-90 metrics into percentile ranks via `QuantileTransformer` and fits Gaussian Mixture Models (GMM) to output hybrid role probabilities, which is mirrored by `train_umap_contextual.py` to calculate positional Z-scores and map non-linear tactical manifolds using UMAP and t-SNE embeddings. Web visualization is driven by `app_dashboard.py` alongside a Next.js web application consuming FastAPI REST endpoints in real time.

---

## 📊 Tactical Archetype Mapping

<div align="center">
  <img src="./data/player_archetypes_pca.png" alt="Tactical Archetypes PCA Map" width="850"/>
  <p><i>Figure 1: Spatial 2D projection using <b>PCA (54.1% Explained Variance)</b> and <b>K-Means Clustering ($K=5$)</b>. Displays tactical progression from defenders on the left (Akanji, Zagadou) to creative wingers/attackers on the right (Sancho, Pulisic), with specialized profiles (Oelschlägel) and extreme statistical outliers isolated into distinct clusters.</i></p>
</div>

---

## 📌 Core Architecture & Machine Learning Modules

* **Data Quality & Validation (`scripts/quality_check.py`)**: Pre-ingestion validation rules ensuring statistical schema compliance and data hygiene prior to SQLite database loading.
* **Percentile Rank Normalization (`QuantileTransformer`)**: Handles skewed football metrics by converting raw per-90 stats into equitable percentile ranks ($0.0$ to $1.0$).
* **Tactical Archetype Profiling (`scripts/train_player_clustering.py`)**: Reduces multi-metric dimensionality to principal components and segments players into $5$ core tactical archetypes via PCA + K-Means.
* **Soft Clustering & Hybrid Roles (`scripts/train_advanced_ml.py`)**: Computes probabilistic multi-cluster membership vectors $\sum_{k=1}^{K} P(\text{Cluster}_k \vert{} \text{Player}) = 1.0$ alongside model confidence scores using Gaussian Mixture Models.
* **Outlier & Gem Scouting (`scripts/detect_scouting_outliers.py`)**: Identifies high-value, unique statistical profiles and undervalued targets operating outside normal performance distributions using Isolation Forests.
* **Contextual Positional Embeddings (`scripts/train_umap_contextual.py`)**: Normalizes performance metrics strictly within positional groups (`position_group`) using Positional Z-Scores and projects non-linear tactical manifolds via UMAP / t-SNE.
* **Intra-Cluster Replacement Search (`scripts/build_player_similarities.py`)**: Cosine similarity engine calculated strictly within tactical clusters to deliver realistic replacement candidates.

---

## 🛠️ Tech Stack

* **Core Engine**: Python 3.10+
* **Database & ORM**: SQLite (`data/football_analytics.db`), SQLAlchemy, Prisma ORM
* **Data Science & ML**: Pandas, NumPy, Scikit-learn, UMAP-learn
* **Visualization**: Matplotlib, Seaborn, Plotly, Streamlit (`app_dashboard.py`)
* **Frontend & Web Backend**: Next.js 14, React, Tailwind CSS, FastAPI REST API

---

## 📁 Repository Structure

```text
football_analytics_project/
├── assests/                         # Static assets & media
├── config/                          # Application & pipeline configuration
├── data/
│   ├── processed/                   # Transformed datasets
│   ├── raw/                         # Ingested raw datasets
│   ├── reference/                   # Mappings & League metadata
│   ├── football_analytics.db        # Core SQLite Database
│   └── player_archetypes_pca.png    # PCA & K-Means Spatial Visualization
├── logs/                            # Pipeline execution & audit logs
├── prisma/                          # Prisma ORM schema & migrations
├── scripts/                         # Modular ML & data processing scripts
├── test/                            # Integration & unit test suites
├── web/                             # Next.js full-stack application
├── .env                             # Environment variables
├── .gitignore                       # Git exclusion rules
├── app_dashboard.py                 # Analytics dashboard interface
├── package.json                     # Node.js project configuration
├── package-lock.json                # Dependency lockfile
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation