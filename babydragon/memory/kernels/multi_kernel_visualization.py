from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
from babydragon.memory.kernels.multi_kernel import MultiKernel
from sklearn.manifold import TSNE


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
            plt.plot(S)
            plt.show()
