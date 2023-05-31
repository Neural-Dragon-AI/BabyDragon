memory_kernel
=============

.. code-block:: python

	
	
	class MemoryKernel(MemoryIndex):
	    def __init__(
	        self,
	        mem_index: MemoryIndex,
	        name: str = "memory_kernel",
	        k: int = 2,
	        save_path: str = None,
	    ):
	        """
	        Initialize the MemoryKernel with a MemoryIndex instance, a name, k value, and save path.
	
	        Args:
	            mem_index (MemoryIndex): A MemoryIndex instance.
	            name (str, optional): The name of the MemoryKernel. Defaults to "memory_kernel".
	            k (int, optional): The number of hops for message passing. Defaults to 2.
	            save_path (str, optional): The path to save the MemoryKernel. Defaults to None.
	        """
	        super().__init__(
	            index=mem_index.index,
	            values=mem_index.values,
	            embeddings=mem_index.embeddings,
	            name=name,
	            save_path=save_path,
	        )
	        self.k = k
	        if len(self.values) > 0: 
	            self.create_k_hop_index(k=k)
	        else:
	            raise ValueError("The input MemoryIndex is empty. Please check the input MemoryIndex.")
	
	    def cos_sim(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
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
	        self,
	        embedding_set: np.ndarray,
	        threshold: float = 0.65,
	        use_softmax: bool = False,
	    ) -> np.ndarray:
	
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
	
	    def k_hop_message_passing(
	        self, A: np.ndarray, node_features: np.ndarray, k: int
	    ) -> Tuple[np.ndarray, np.ndarray]:
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
	
	    def graph_sylvester_embedding(self, G: Tuple, m: int, ts: np.ndarray) -> np.ndarray:
	        """
	        Compute the spectral kernel descriptor or the Spectral Graph Wavelet descriptor.
	
	        Args:
	            G (Tuple): A tuple containing the graph's vertices (V) and weights (W).
	            m (int): The number of singular values to consider.
	            ts (np.ndarray): The spectral scales.
	
	        Returns:
	            np.ndarray: The node_embeddings matrix.
	        """
	        V, W = G
	        n = len(V)
	        D_BE = np.diag(W.sum(axis=1))
	        L_BE = np.identity(n) - np.dot(
	            np.diag(1 / np.sqrt(D_BE.diagonal())),
	            np.dot(W, np.diag(1 / np.sqrt(D_BE.diagonal()))),
	        )
	
	        A = W
	        B = L_BE
	        C = np.identity(n)
	        X = solve_sylvester(A, B, C)
	
	        U, S, _ = svd(X, full_matrices=False)
	        U_m = U[:, :m]
	        S_m = S[:m]
	
	        node_embeddings = np.zeros((n, m))
	
	        for i in range(n):
	            for s in range(m):
	                # Spectral kernel descriptor
	                node_embeddings[i, s] = np.exp(-ts[s] * S_m[s]) * U_m[i, s]
	
	        return node_embeddings
	
	    def gen_gse_embeddings(
	        self, A: np.ndarray, embeddings: np.ndarray, m: int = 7
	    ) -> np.ndarray:
	        """
	        Generate Graph Sylvester Embeddings.
	
	        Args:
	            A (np.ndarray): The adjacency matrix of the graph.
	            embeddings (np.ndarray): The original node embeddings.
	            m (int, optional): The number of spectral scales. Defaults to 7.
	
	        Returns:
	            np.ndarray: The generated Graph Sylvester Embeddings.
	        """
	        V = list(range(len(embeddings)))
	        W = A
	
	        G = (V, W)
	        ts = np.linspace(0, 1, m)  # equally spaced scales
	
	        gse_embeddings = self.graph_sylvester_embedding(G, m, ts)
	        return gse_embeddings
	
	    def create_k_hop_index(self, k: int = 2):
	        """
	        Create a k-hop index by computing the adjacency matrix, k-hop adjacency matrix,
	        aggregated features, and updating the memory index.
	
	        Args:
	            k (int, optional): The number of hops for message passing. Defaults to 2.
	        """
	        self.k = k
	        print("Computing the adjacency matrix")
	        print("Embeddings shape: ", self.embeddings.shape)
	        self.A = self.compute_kernel(self.embeddings, threshold=0.65, use_softmax=False)
	        print("Computing the k-hop adjacency matrix and aggregated features")
	        self.A_k, self.node_embeddings = self.k_hop_message_passing(
	            self.A, self.embeddings, k
	        )
	        print("Updating the memory index")
	        self.k_hop_index = MemoryIndex(name=self.name)
	        self.k_hop_index.init_index(values=self.values, embeddings=self.node_embeddings)
	
	    @classmethod
	    def from_task_results(cls, task_memory_index):
	        new_memory_kernel = cls(mem_index=task_memory_index)
	
	        # Create a new index for the new MemoryKernel
	        new_memory_kernel.create_k_hop_index()
	
	        return new_memory_kernel
	

.. automodule:: memory_kernel
   :members:
