from sklearn.cluster import SpectralClustering
import hdbscan
import numpy as np
from typing import List

class ClusterPaths:
    def create_paths(self, embeddings: np.ndarray, num_clusters: int) -> List[List[int]]:
        raise NotImplementedError

class HDBSCANPaths(ClusterPaths):
    def create_paths(self, embeddings: np.ndarray, num_clusters: int) -> List[List[int]]:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clusters)
        cluster_assignments = clusterer.fit_predict(embeddings)
        paths = [[] for _ in range(num_clusters)]
        for i, cluster in enumerate(cluster_assignments):
            paths[cluster].append(i)
        paths = [path for path in paths if path]
        return paths

class SpectralClusteringPaths(ClusterPaths):
    def create_paths(self, embeddings: np.ndarray, num_clusters: int) -> List[List[int]]:
        spectral_clustering = SpectralClustering(
            n_clusters=num_clusters, affinity="nearest_neighbors", random_state=42
        )
        cluster_assignments = spectral_clustering.fit_predict(embeddings)
        paths = [[] for _ in range(num_clusters)]
        for i, cluster in enumerate(cluster_assignments):
            paths[cluster].append(i)
        paths = [path for path in paths if path]
        return paths
