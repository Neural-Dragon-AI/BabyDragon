from typing import List, Tuple
from babydragon.memory.indexes.memory_index import MemoryIndex
import numpy as np
from sklearn.cluster import SpectralClustering
from babydragon.tasks.llm_task import LLMWriter
import hdbscan
import umap.umap_ as umap
from tqdm import tqdm
import itertools
from scipy.linalg import solve_sylvester
from numpy.linalg import svd
from babydragon.chat.chat import Chat
from sklearn.metrics import normalized_mutual_info_score
import scipy



class MemoryKernel(MemoryIndex):
    def __init__(self, mem_index, name="memory_kernel", k=2, save_path=None):
        super().__init__(index = mem_index.index, values=mem_index.values, embeddings=mem_index.embeddings, name=name, save_path=save_path)
        self.k = k
        self.create_k_hop_index(k=k)

    def cos_sim(self, a, b):
        """
        Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
        :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
        """
        if not isinstance(a, np.ndarray):
            a = np.array(a)

        if not isinstance(b, np.ndarray):
            b = np.array(b)

        if len(a.shape) == 1:
            a = a[np.newaxis, :]

        if len(b.shape) == 1:
            b = b[np.newaxis, :]

        a_norm = a / np.linalg.norm(a, ord=2, axis=1, keepdims=True)
        b_norm = b / np.linalg.norm(b, ord=2, axis=1, keepdims=True)
        return np.dot(a_norm, b_norm.T)

    def compute_kernel(
        self, embedding_set, threshold=0.65, use_softmax=False, cos_sim_batch=True
    ):
        """
        Compute the adjacency matrix of the graph.

        Parameters:
        embedding_set (numpy array): The embedding matrix of the nodes.
        threshold (float): The threshold for the adjacency matrix.
        use_softmax (bool): Whether to use softmax to compute the adjacency matrix.
        cos_sim_batch (bool): Whether to use batch processing to compute the cosine similarity.

        Returns:
        adj_matrix (numpy array): The adjacency matrix of the graph.
        """

        A = self.cos_sim(embedding_set, embedding_set)
        if use_softmax:
            # softmax
            A = np.exp(A)
            A = A / np.sum(A, axis=1)[:, np.newaxis]
        adj_matrix = np.zeros_like(A)
        adj_matrix[A > threshold] = 1
        adj_matrix[A <= threshold] = 0
        adj_matrix = adj_matrix.astype(np.float32)
        return adj_matrix

    def k_hop_message_passing(self, A, node_features, k):
        """
        Compute the k-hop adjacency matrix and aggregated features using message passing.

        Parameters:
        A (numpy array): The adjacency matrix of the graph.
        node_features (numpy array): The feature matrix of the nodes.
        k (int): The number of hops for message passing.

        Returns:
        A_k (numpy array): The k-hop adjacency matrix.
        agg_features (numpy array): The aggregated feature matrix for each node in the k-hop neighborhood.
        """

        print("Compute the k-hop adjacency matrix")
        A_k = np.linalg.matrix_power(A, k)

        print("Aggregate the messages from the k-hop neighborhood:")
        agg_features = node_features.copy()

        for i in tqdm(range(k)):
            agg_features += np.matmul(np.linalg.matrix_power(A, i + 1), node_features)

        return A_k, agg_features

    def graph_sylvester_embedding(self, G, m, ts):
        V, W = G
        n = len(V)
        # Step 2: Compute L_BE
        D_BE = np.diag(W.sum(axis=1))
        L_BE = np.identity(n) - np.dot(np.diag(1 / np.sqrt(D_BE.diagonal())), np.dot(W, np.diag(1 / np.sqrt(D_BE.diagonal()))))

        # Step 3: Solve the discrete-time Sylvester equation
        A = W
        B = L_BE
        C = np.identity(n)
        X = solve_sylvester(A, B, C)

        # Step 4: Compute the largest m singular values and associated singular vectors of X
        U, S, Vh = svd(X, full_matrices=False)
        U_m = U[:, :m]
        S_m = S[:m]

        # Step 5: Compute the spectral kernel descriptor or the Spectral Graph Wavelet descriptor
        node_embeddings = np.zeros((n, m))

        for i in range(n):
            for s in range(m):
                # Spectral kernel descriptor
                node_embeddings[i, s] = np.exp(-ts[s] * S_m[s]) * U_m[i, s]

        return node_embeddings

    def gen_gse_embeddings(self, A, embeddings, m: int = 7):
        V = list(range(len(embeddings)))
        W = A

        G = (V, W)
        ts = np.linspace(0, 1, m)  # equally spaced scales

        gse_embeddings = self.graph_sylvester_embedding(G, m, ts)
        return gse_embeddings

    def create_k_hop_index(self, k=2):
        print("Computing the adjacency matrix")
        print("Embeddings shape: ", self.embeddings.shape)
        self.A = self.compute_kernel(self.embeddings, threshold=0.65, use_softmax=False)
        print("Computing the k-hop adjacency matrix and aggregated features")
        self.A_k, self.node_embeddings = self.k_hop_message_passing(
            self.A, self.embeddings, k
        )
        print("Updating the memory index")
        self.k_hop_index = MemoryIndex( name=self.name)
        self.k_hop_index.init_index(values=self.values, embeddings=self.node_embeddings)


class MemoryKernelGroup(MemoryKernel):
    def __init__(self, memory_kernel_dict: dict, name="memory_kernel_group"):
        self.memory_kernel_dict = memory_kernel_dict
        self.name = name

    def create_paths_hdbscan(self, embeddings: np.ndarray, num_clusters: int) -> List[List[int]]:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clusters)
        cluster_assignments = clusterer.fit_predict(embeddings)

        paths = [[] for _ in range(num_clusters)]
        for i, cluster in enumerate(cluster_assignments):
            paths[cluster].append(i)

        return paths

    def create_paths_spectral_clustering(self, embeddings: np.ndarray, num_clusters: int) -> List[List[int]]:
        spectral_clustering = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', random_state=42)
        cluster_assignments = spectral_clustering.fit_predict(embeddings)

        paths = [[] for _ in range(num_clusters)]
        for i, cluster in enumerate(cluster_assignments):
            paths[cluster].append(i)

        return paths

    def calc_shgo_mode(self, scores):
        def objective(x):
            return -self.estimate_pdf(scores)(x)

        bounds = [(min(scores), max(scores))]
        result = scipy.optimize.shgo(objective, bounds)
        return result.x

    def estimate_pdf(self, scores):
        pdf = scipy.stats.gaussian_kde(scores)
        return pdf

    def sort_paths_by_mode_distance(self, kernel_label):
        paths = self.path_group[kernel_label]
        memory_kernel = self.memory_kernel_dict[kernel_label]
        sorted_paths = []
        for path in paths:
            cluster_embeddings = [memory_kernel.node_embeddings[i] for i in path]
            cluster_embeddings = np.array(cluster_embeddings)
            cluster_mean = np.mean(cluster_embeddings, axis=0)
            scores = [(i, np.linalg.norm(cluster_mean - emb)) for i, emb in zip(path, cluster_embeddings)]
            mu = self.calc_shgo_mode(scores)
            sigma = np.std(scores)
            scores = [(i, np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))) for i, x in scores]
            #sorth path by score
            sorted_path_and_scores = sorted(scores, key=lambda x: x[1], reverse=True)
            sorted_path = [x[0] for x in sorted_path_and_scores]
            sorted_paths.append(sorted_path)
        self.path_group[kernel_label] = sorted_paths

    def gen_aligned_kernel(self, chatbot:Chat, parent_kernel_label: str, child_kernel_label: str):
        llm_writer = LLMWriter(index=self.memory_kernel_dict[parent_kernel_label], path=self.path_group[parent_kernel_label], chatbot=chatbot, write_func=None, max_workers=1)
        new_index = llm_writer.write()
        new_memory_kernel = MemoryKernel(mem_index = new_index, name=child_kernel_label)
        new_memory_kernel.create_k_hop_index(k=2)
        self.memory_kernel_dict[child_kernel_label] = new_memory_kernel


    def generate_path_groups(self, method="hdbscan"):
        path_group = {}
        for k, v in self.memory_kernel_dict.items():
            embeddings = v.node_embeddings
            num_clusters = int(np.sqrt(len(embeddings)))
            if method == "hdbscan":
                paths = self.create_paths_hdbscan(embeddings, num_clusters)
            elif method == "spectral_clustering":
                paths = self.create_paths_spectral_clustering(embeddings, num_clusters)
            path_group[k] = paths

        self.path_group = path_group

    def batch_sort_kernel_group(self, kernel_label: str):
        #if all kernels are of the same dimensions, then we can sort them all at once using the paths of the given key
        if all([v.node_embeddings.shape == self.memory_kernel_dict[kernel_label].node_embeddings.shape for k, v in self.memory_kernel_dict.items()]):
            self.memory_kernel_sort(self.path_group[kernel_label])
        else:
            return ValueError("Not all kernels are of the same dimensions.")

    def memory_kernel_sort(self, paths: List[List[int]]):
        pass

    def is_kernel_group_isomorphic(self):
        pass

class MemoryKernelGroupStabilityAnalysis:
    def __init__(self, memory_kernel_group: MemoryKernelGroup):
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