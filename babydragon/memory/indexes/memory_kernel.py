from babydragon.memory.indexes.memory_index import MemoryIndex
from babydragon.working_memory.associative_memory.probability_density_functions import calc_shgo_mode, estimate_pdf
from babydragon.working_memory.associative_memory.group_by_rank import group_items_by_rank_buckets_svd
#from babydragon.working_memory.associative_memory.nmi import run_stability_analysis
import numpy as np
import faiss
import pandas as pd
from tqdm import tqdm


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


class MemoryKernelGroup:
    def __init__(self, memory_kernel_dict: dict, name="memory_kernel_group", save_path=None):
        self.memory_kernel_dict = memory_kernel_dict

    def rank_decomp_and_merge(self, component_window_size=1, threshold=0.13):
        bucket_groups = {}
        for key, mem_kernel in self.memory_kernel_dict.items():
            print(key)
            code_values = mem_kernel.values
            print(f'code_values: {len(code_values)}')
            code_embeddings = mem_kernel.node_embeddings
            print(f'code_embeddings: {code_embeddings.shape}')
            _, _, VT = np.linalg.svd(mem_kernel.A_k)
            code_df = pd.DataFrame(code_values, columns=['code'])
            rank_buckets = group_items_by_rank_buckets_svd(code_df, code_embeddings, VT, num_components=component_window_size, threshold=threshold,use_softmax = False)
            bucket_groups[key] = rank_buckets
        return bucket_groups