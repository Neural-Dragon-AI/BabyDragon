from babydragon.memory.indexes.memory_index import MemoryIndex
import numpy as np
from tqdm import tqdm


class MemoryKernel(MemoryIndex):
    def __init__(self, values, embeddings, name="memory_kernel", save_path=None):
        super().__init__(values, embeddings, name, save_path)
        self.create_k_hop_index()
        
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

    def cos_sim_batch(self, a: np.ndarray, b: np.ndarray, batch_size: int = 128):
        """
        Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j using batch processing.
        :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
        """

        if not isinstance(a, np.ndarray):
            a = np.array(a)

        if not isinstance(b, np.ndarray):
            b = np.array(b)

        if len(a.shape) == 1:
            a = np.expand_dims(a, 0)

        if len(b.shape) == 1:
            b = np.expand_dims(b, 0)

        a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)

        sim_matrix = []
        for i in range(0, len(a_norm), batch_size):
            a_batch = a_norm[i : i + batch_size]
            sim_batch = np.matmul(a_batch, b_norm.T)
            sim_matrix.append(sim_batch)
        sim_matrix = np.concatenate(sim_matrix, axis=0)
        return sim_matrix

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
        if cos_sim_batch:
            A = self.cos_sim_batch(embedding_set, embedding_set)
        else:
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

    def create_k_hop_index(self, k):
        print("Computing the adjacency matrix")
        print("Embeddings shape: ", self.embeddings.shape)
        self.A = self.compute_kernel(self.embeddings, threshold=0.65, use_softmax=False)
        print("Computing the k-hop adjacency matrix and aggregated features")
        self.A_k, self.node_embeddings = self.k_hop_message_passing(
            self.A, self.embeddings, k
        )
        print("Updating the memory index")
        self.k_hop_index = MemoryIndex(index=None, values=self.values, embeddings=self.node_embeddings, name=self.memory_index.name)
  