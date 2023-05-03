import numpy as np
from scipy.spatial.distance import cosine
from sklearn.neighbors import KernelDensity
import scipy
from typing import List

def calc_shgo_mode(scores: List[float]) -> float:
    def objective(x):
        return -estimate_pdf(scores)(x)

    bounds = [(min(scores), max(scores))]
    result = scipy.optimize.shgo(objective, bounds)
    return result.x

def estimate_pdf(scores: List[float]) -> callable:
    pdf = scipy.stats.gaussian_kde(scores)
    return pdf

def sort_paths_by_mode_distance(paths, memory_kernel, distance_metric: str = "cosine") -> List[List[int]]:
    sorted_paths = []
    for i, path in enumerate(paths):
        cluster_embeddings = [memory_kernel.node_embeddings[i] for i in path]
        cluster_embeddings = np.array(cluster_embeddings)
        cluster_mean = np.mean(cluster_embeddings, axis=0)
        if distance_metric == "cosine" or distance_metric == "guassian":
            scores = [
                (i, cosine(cluster_mean, emb))
                for i, emb in zip(path, cluster_embeddings)
            ]
        elif distance_metric == "euclidean":
            scores = [
                (i, np.linalg.norm(cluster_mean - emb))
                for i, emb in zip(path, cluster_embeddings)
            ]
        score_values = [score for _, score in scores]  # Extract score values
        mu = calc_shgo_mode(score_values)
        sigma = np.std(score_values)
        if distance_metric == "guassian":
            scores = [
                (i, np.exp(-((x - mu) ** 2) / (2 * sigma**2))) for i, x in scores
            ]
        # Sort path by score
        sorted_path_and_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        sorted_path = [x[0] for x in sorted_path_and_scores]
        sorted_paths.append(sorted_path)
    return sorted_paths

def sort_paths_by_kernel_density(paths, memory_kernel, distance_metric: str = "cosine") -> List[List[int]]:
    sorted_paths = []
    for i, path in enumerate(paths):
        cluster_embeddings = [memory_kernel.node_embeddings[i] for i in path]
        cluster_embeddings = np.array(cluster_embeddings)
        cluster_mean = np.mean(cluster_embeddings, axis=0)
        if distance_metric == "cosine":
            scores = [
                (i, cosine(cluster_mean, emb))
                for i, emb in zip(path, cluster_embeddings)
            ]
        elif distance_metric == "euclidean":
            scores = [
                (i, np.linalg.norm(cluster_mean - emb))
                for i, emb in zip(path, cluster_embeddings)
            ]
        score_values = [score for _, score in scores]  # Extract score values

        # Estimate PDF using Kernel Density Estimation
        kde = KernelDensity(kernel="gaussian", bandwidth=0.2).fit(
            np.array(score_values).reshape(-1, 1)
        )
        kde_scores = [kde.score_samples([[x]])[0] for _, x in scores]

        # Sort path by score
        sorted_path_and_scores = sorted(
            zip(path, kde_scores), key=lambda x: x[1], reverse=True
        )
        sorted_path = [x[0] for x in sorted_path_and_scores]
        sorted_paths.append(sorted_path)
    return sorted_paths
