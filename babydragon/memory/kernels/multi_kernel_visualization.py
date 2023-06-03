from matplotlib import pyplot as plt
from typing import Tuple
import matplotlib.cm as cm
import numpy as np
from babydragon.memory.kernels.multi_kernel import MultiKernel
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score
import itertools

class MultiKernelVisualization:
    def __init__(self, memory_kernel_group: MultiKernel):
        self.memory_kernel_group = memory_kernel_group
        self.memory_kernel_dict = memory_kernel_group.memory_kernel_dict
        self.memory_kernel_group.generate_path_groups()

    def plot_embeddings_with_path(self, embeddings, title, paths):
        tsne = TSNE(n_components=2, random_state=42)
        reduced_embeddings = tsne.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        colors = cm.rainbow(np.linspace(0, 1, len(paths)))
        for i, path in enumerate(paths):
            path_embeddings = reduced_embeddings[path]
            plt.scatter(
                path_embeddings[:, 0],
                path_embeddings[:, 1],
                color=colors[i],
                label=f"Cluster {i}",
            )
            for j in range(len(path) - 1):
                plt.plot(
                    [path_embeddings[j, 0], path_embeddings[j + 1, 0]],
                    [path_embeddings[j, 1], path_embeddings[j + 1, 1]],
                    color=colors[i],
                )
        plt.title(title)
        plt.legend()
        plt.show()

    def visualize_paths(self):
        #loop through memory kernels and print path_group
        for key, kernel in self.memory_kernel_dict.items():
            print(f"Kernel: {key}")
            paths = self.memory_kernel_group.path_group[key]
            print(f"Path Group: {paths}")
            node_embeddings = kernel.node_embeddings
            self.plot_embeddings_with_path(
                node_embeddings, f"Node Embeddings for {key}", paths
            )
    def plot_singular_values(self):
        #loop through memory kernels and print path_group
        for key, kernel in self.memory_kernel_dict.items():
            print(f"Kernel: {key}")
            A_k = kernel.A_k
            U, S, V = np.linalg.svd(A_k)
            plt.plot(np.log(S))
            plt.show()

class MultiKernelStabilityAnalysis:
    def __init__(self, memory_kernel_group: MultiKernel):
        self.memory_kernel_group = memory_kernel_group

    def get_cluster_labels(self, kernel_label: str) -> Tuple[np.ndarray, int]:
        paths = self.memory_kernel_group.path_group[kernel_label]
        num_clusters = len(paths)
        cluster_labels = np.empty(len(self.memory_kernel_group.memory_kernel_dict[kernel_label].node_embeddings), dtype=int)

        for cluster_index, path in enumerate(paths):
            cluster_labels[path] = cluster_index

        return cluster_labels, num_clusters

    def compute_nmi(self, kernel_label1: str, kernel_label2: str) -> float:
        cluster_labels1, _ = self.get_cluster_labels(kernel_label1)
        cluster_labels2, _ = self.get_cluster_labels(kernel_label2)
        nmi = normalized_mutual_info_score(cluster_labels1, cluster_labels2)
        return nmi

    def evaluate_stability(self) -> float:
        kernel_labels = list(self.memory_kernel_group.memory_kernel_dict.keys())
        pairwise_combinations = list(itertools.combinations(kernel_labels, 2))
        nmi_sum = 0

        for kernel_label1, kernel_label2 in pairwise_combinations:
            nmi = self.compute_nmi(kernel_label1, kernel_label2)
            nmi_sum += nmi

        stability_score = nmi_sum / len(pairwise_combinations)
        return stability_score