"""Odysseus Unsupervised Clustering & Behavioral Segmentation Engine.

Performs optimal K-Means, Agglomerative Hierarchical clustering,
and 2D/3D PCA latent space projections to profile regulatory risk archetypes.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config.settings import MODELS_DIR, PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


class SupervisoryClusterEngine:
    """Unsupervised segmentation of regulatory periods and supervisory risk archetypes."""

    def __init__(self, models_dir: Path = MODELS_DIR, random_state: int = 42):
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca_3d = PCA(n_components=3, random_state=random_state)
        self.kmeans_model: Optional[KMeans] = None
        self.feature_names_: List[str] = []

    def fit_predict_clusters(
        self,
        df: Optional[pd.DataFrame] = None,
        k_range: Tuple[int, int] = (2, 5),
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Find optimal cluster count k and segment supervisory data."""
        if df is None:
            data_path = PROCESSED_DATA_DIR / "features_supervision_ews.parquet"
            if not data_path.exists():
                raise FileNotFoundError(f"Feature dataset not found at {data_path}")
            df = pd.read_parquet(data_path)

        drop_cols = ["year", "quarter", "period", "target_high_risk_period", "supervisory_risk_index"]
        self.feature_names_ = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]

        X_raw = df[self.feature_names_].values
        X_scaled = self.scaler.fit_transform(X_raw)

        logger.info(f"Evaluating optimal clustering on {len(df)} samples across {len(self.feature_names_)} features...")

        # 1. Search for optimal k
        best_k = 3
        best_silhouette = -1.0
        k_evaluations = []

        max_k = min(k_range[1], len(df) - 1)
        min_k = min(k_range[0], max_k)

        for k in range(min_k, max_k + 1):
            km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
            labels = km.fit_predict(X_scaled)

            if len(set(labels)) > 1:
                sil = float(silhouette_score(X_scaled, labels))
                db = float(davies_bouldin_score(X_scaled, labels))
                ch = float(calinski_harabasz_score(X_scaled, labels))
            else:
                sil, db, ch = -1.0, 99.0, 0.0

            k_evaluations.append({
                "k": k,
                "silhouette_score": round(sil, 4),
                "davies_bouldin_index": round(db, 4),
                "calinski_harabasz_score": round(ch, 2),
            })

            if sil > best_silhouette:
                best_silhouette = sil
                best_k = k

        logger.info(f"Selected Optimal Clusters k={best_k} (Silhouette Score: {best_silhouette:.4f})")

        # 2. Fit Final Models (KMeans & Hierarchical)
        self.kmeans_model = KMeans(n_clusters=best_k, n_init=20, random_state=self.random_state)
        cluster_labels_kmeans = self.kmeans_model.fit_predict(X_scaled)

        agg = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
        cluster_labels_hierarchical = agg.fit_predict(X_scaled)

        # 3. Dimensionality Reduction (PCA 3D & 2D)
        pca_coords = self.pca_3d.fit_transform(X_scaled)

        # 4. Construct Output Dataframe
        result_df = df.copy()
        result_df["cluster_id"] = cluster_labels_kmeans
        result_df["cluster_hierarchical"] = cluster_labels_hierarchical
        result_df["pca_1"] = np.round(pca_coords[:, 0], 3)
        result_df["pca_2"] = np.round(pca_coords[:, 1], 3)
        result_df["pca_3"] = np.round(pca_coords[:, 2], 3)

        # 5. Define Cluster Archetype Profiles
        cluster_profiles = []
        cluster_mean_risk = result_df.groupby("cluster_id")["total_sanciones_impuestas"].mean()
        risk_rank = cluster_mean_risk.rank().to_dict()

        archetype_map = {
            1: "Arquetipo I: Supervision Estable / Baja Incidencia",
            2: "Arquetipo II: Vigilancia Regular / Conducta Moderada",
            3: "Arquetipo III: Alta Intensidad Sancionadora / Friccion",
            4: "Arquetipo IV: Alerta Critica / Regularizacion Activa"
        }

        result_df["cluster_archetype"] = result_df["cluster_id"].apply(
            lambda c: archetype_map.get(int(risk_rank.get(c, 1)), f"Cluster {c}")
        )

        for c_id in range(best_k):
            sub = result_df[result_df["cluster_id"] == c_id]
            cluster_profiles.append({
                "cluster_id": int(c_id),
                "archetype": archetype_map.get(int(risk_rank.get(c_id, 1)), f"Cluster {c_id}"),
                "size": len(sub),
                "pct_of_total": round(100.0 * len(sub) / len(result_df), 1),
                "mean_sanciones": round(float(sub["total_sanciones_impuestas"].mean()), 2),
                "mean_infracciones": round(float(sub["total_infracciones_imputadas"].mean()), 2),
                "mean_monto_dop": round(float(sub["monto_sanciones_dop"].mean()), 2),
                "mean_solicitudes_aml": round(float(sub["total_solicitudes_aml"].mean()), 2),
            })

        # Save metadata and artifacts
        meta = {
            "task": "supervisory_clustering",
            "optimal_k": best_k,
            "best_silhouette": round(best_silhouette, 4),
            "k_evaluations": k_evaluations,
            "cluster_profiles": cluster_profiles,
            "pca_explained_variance_ratio": [float(x) for x in self.pca_3d.explained_variance_ratio_],
            "total_explained_variance_3d": round(float(np.sum(self.pca_3d.explained_variance_ratio_)), 4),
        }

        with open(self.models_dir / "clustering_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        joblib.dump(self, self.models_dir / "supervisory_clustering_engine.joblib")
        result_df.to_parquet(self.models_dir / "supervisory_clustered.parquet", index=False)

        logger.info(f"Clustering Complete | Saved to {self.models_dir / 'supervisory_clustered.parquet'}")
        return result_df, meta


def run_clustering_pipeline():
    """Execute unsupervised clustering pipeline."""
    engine = SupervisoryClusterEngine()
    df_clustered, meta = engine.fit_predict_clusters()
    print(f">>> [Clustering] Optimal clusters: k={meta['optimal_k']} (Silhouette: {meta['best_silhouette']})", flush=True)
    print(">>> [Clustering] Cluster Profiles:")
    for cp in meta["cluster_profiles"]:
        print(f"  * {cp['archetype']}: {cp['size']} trimestres ({cp['pct_of_total']}%) | Sanciones Media={cp['mean_sanciones']} | Infracciones Media={cp['mean_infracciones']}", flush=True)


if __name__ == "__main__":
    run_clustering_pipeline()
