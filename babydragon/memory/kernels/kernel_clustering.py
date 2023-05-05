from typing import List

import hdbscan
import numpy as np
from sklearn.cluster import SpectralClustering


class ClusterPaths:
    def create_paths(
        self, embeddings: np.ndarray, num_clusters: int
    ) -> List[List[int]]:
        raise NotImplementedError


class HDBSCANPaths(ClusterPaths):
    def create_paths(
        self, embeddings: np.ndarray, num_clusters: int
    ) -> List[List[int]]:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clusters)
        cluster_assignments = clusterer.fit_predict(embeddings)
        paths = [[] for _ in range(num_clusters)]
        for i, cluster in enumerate(cluster_assignments):
            paths[cluster].append(i)
        paths = [path for path in paths if path]
        return paths


class SpectralClusteringPaths(ClusterPaths):
    def create_paths(
        self, A: np.ndarray, num_clusters: int
    ) -> List[List[int]]:
        n_samples = A.shape[0]
        n_neighbors = min(n_samples - 1, 10)  # Set n_neighbors to min(n_samples - 1, 10)
        spectral_clustering = SpectralClustering(
            n_clusters=num_clusters,
            affinity="precomputed",
            n_neighbors=n_neighbors,
            random_state=42,
        )
        cluster_assignments = spectral_clustering.fit_predict(A)
        paths = [[] for _ in range(num_clusters)]
        for i, cluster in enumerate(cluster_assignments):
            paths[cluster].append(i)
        paths = [path for path in paths if path]
        return paths
