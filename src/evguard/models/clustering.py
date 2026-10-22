"""K-Means clustering on battery operational cycles and degradation profiles."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from evguard.utils.config import (
    FEATURE_COLUMNS,
    KMEANS_METRICS_FILE,
    KMEANS_MODEL_FILE,
    KMEANS_N_CLUSTERS,
    KMEANS_RANDOM_STATE,
)
from evguard.utils.logger import get_logger

logger = get_logger(__name__)


class BatteryClusterModel:
    """Standardized K-Means clustering model for battery degradation cycle profiling."""

    def __init__(
        self,
        n_clusters: int = KMEANS_N_CLUSTERS,
        random_state: int = KMEANS_RANDOM_STATE,
        features: list[str] | None = None,
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.features = features or FEATURE_COLUMNS
        self.scaler = StandardScaler()
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300,
        )
        self.cluster_labels_map: dict[int, str] = {}
        self.evaluation_metrics: dict = {}
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "BatteryClusterModel":
        """Fit scaler and KMeans clustering on plausible battery records."""
        # Use plausible records for clean model fitting
        train_df = df[df.get("is_plausible", True) == True].copy()  # noqa: E712
        available_features = [f for f in self.features if f in train_df.columns]

        if len(available_features) < 2:
            raise ValueError(f"Insufficient features available for clustering: {available_features}")

        self.features = available_features
        X = train_df[self.features].dropna()

        if len(X) < self.n_clusters:
            raise ValueError(f"Dataset has fewer rows ({len(X)}) than clusters ({self.n_clusters})")

        logger.info(f"Fitting KMeans (k={self.n_clusters}) on {len(X)} records using features: {self.features}")

        X_scaled = self.scaler.fit_transform(X)
        self.kmeans.fit(X_scaled)

        # Calculate validation metrics
        inertia = float(self.kmeans.inertia_)
        sil_score = float(silhouette_score(X_scaled, self.kmeans.labels_)) if len(X) > self.n_clusters else 0.0
        db_score = float(davies_bouldin_score(X_scaled, self.kmeans.labels_)) if len(X) > self.n_clusters else 0.0

        # Compute cluster centroids and assign domain-grounded engineering labels
        centroids = pd.DataFrame(
            self.scaler.inverse_transform(self.kmeans.cluster_centers_),
            columns=self.features,
        )

        self._assign_cluster_labels(centroids)

        self.evaluation_metrics = {
            "n_clusters": self.n_clusters,
            "inertia": round(inertia, 2),
            "silhouette_score": round(sil_score, 4),
            "davies_bouldin_index": round(db_score, 4),
            "features_used": self.features,
            "cluster_profiles": centroids.round(3).to_dict(orient="index"),
            "cluster_labels": self.cluster_labels_map,
            "samples_trained": len(X),
        }
        self.is_fitted = True
        logger.info(
            f"KMeans fitting complete: Silhouette={sil_score:.4f}, "
            f"Davies-Bouldin={db_score:.4f}, Inertia={inertia:.2f}"
        )
        return self

    def _assign_cluster_labels(self, centroids: pd.DataFrame) -> None:
        """Assign scientifically sound, conservative engineering labels to each cluster based on SOH and resistance."""
        labels = {}
        # Rank clusters by centroid SOH
        soh_col = "soh_pct" if "soh_pct" in centroids.columns else centroids.columns[0]
        sorted_indices = centroids[soh_col].sort_values(ascending=False).index.tolist()

        if len(sorted_indices) == 3:
            labels[sorted_indices[0]] = "Tier 1: Nominal Health & Low Impedance"
            labels[sorted_indices[1]] = "Tier 2: Moderate Thermal / Wear Aging"
            labels[sorted_indices[2]] = "Tier 3: Degraded Capacity / Near End-of-Life"
        else:
            for rank, idx in enumerate(sorted_indices):
                labels[idx] = f"Cluster {idx} (Rank {rank + 1} SOH)"

        self.cluster_labels_map = labels

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign cluster IDs and descriptive labels to input DataFrame."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict() is called.")

        res = df.copy()
        X = res[self.features].fillna(res[self.features].mean())
        X_scaled = self.scaler.transform(X)

        cluster_ids = self.kmeans.predict(X_scaled)
        res["cluster_id"] = cluster_ids
        res["cluster_label"] = [self.cluster_labels_map.get(cid, f"Cluster {cid}") for cid in cluster_ids]
        return res

    def compute_elbow_curve(self, df: pd.DataFrame, max_k: int = 6) -> dict[int, float]:
        """Compute inertia values across k=2 to max_k for elbow-method visualization."""
        train_df = df[df.get("is_plausible", True) == True].copy()  # noqa: E712
        X = train_df[self.features].dropna()
        X_scaled = self.scaler.transform(X)

        inertias = {}
        for k in range(2, min(max_k + 1, len(X))):
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=5)
            km.fit(X_scaled)
            inertias[k] = round(float(km.inertia_), 2)
        return inertias

    def save(
        self,
        model_path: Path = KMEANS_MODEL_FILE,
        metrics_path: Path = KMEANS_METRICS_FILE,
    ) -> None:
        """Serialize model artifact and evaluation metrics to disk."""
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, model_path)

        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.evaluation_metrics, f, indent=2)

        logger.info(f"Model successfully saved to {model_path} and metrics to {metrics_path}")

    @classmethod
    def load(cls, model_path: Path = KMEANS_MODEL_FILE) -> "BatteryClusterModel":
        """Load serialized model artifact from disk."""
        if not model_path.exists():
            raise FileNotFoundError(f"Artifact not found: {model_path}")
        logger.info(f"Loading KMeans model from {model_path}")
        return joblib.load(model_path)


def train_and_evaluate_clustering(df: pd.DataFrame) -> tuple[BatteryClusterModel, pd.DataFrame, dict]:
    """Train K-Means clustering, evaluate metrics, save artifacts, and annotate dataframe."""
    model = BatteryClusterModel(n_clusters=3, random_state=42)
    model.fit(df)
    annotated_df = model.predict(df)
    model.save()
    return model, annotated_df, model.evaluation_metrics
