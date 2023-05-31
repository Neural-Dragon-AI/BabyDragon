multi_kernel
============

.. code-block:: python

	
	
	class MultiKernel(MemoryKernel):
	    def __init__(
	        self,
	        memory_kernel_dict: Dict[str, MemoryKernel],
	        name: str = "memory_kernel_group",
	    ):
	        """
	        Initialize the MultiKernel with a dictionary of MemoryKernel instances.
	
	        Args:
	            memory_kernel_dict (Dict[str, MemoryKernel]): A dictionary of MemoryKernel instances.
	            name (str, optional): The name of the MultiKernel. Defaults to "memory_kernel_group".
	        """
	        self.memory_kernel_dict = memory_kernel_dict
	        self.path_group = {}
	        self.name = name
	

.. code-block:: python

	
	
	class HDBSCANMultiKernel(MultiKernel):
	    def __init__(
	        self,
	        memory_kernel_dict: Dict[str, MemoryKernel],
	        name: str = "memory_kernel_group",
	    ):
	        super().__init__(memory_kernel_dict, name)
	        self.cluster_paths = HDBSCANPaths()
	
	    def generate_path_groups(self, num_clusters: int = None) -> None:
	        path_group = {}
	        for k, v in self.memory_kernel_dict.items():
	            embeddings = v.node_embeddings
	            if num_clusters is None:
	                num_clusters = int(np.sqrt(len(embeddings)))
	            paths = self.cluster_paths.create_paths(embeddings, num_clusters)
	            path_group[k] = paths
	        self.path_group = path_group
	

.. code-block:: python

	
	
	class SpectralClusteringMultiKernel(MultiKernel):
	    def __init__(
	        self,
	        memory_kernel_dict: Dict[str, MemoryKernel],
	        name: str = "memory_kernel_group",
	    ):
	        super().__init__(memory_kernel_dict, name)
	        self.cluster_paths = SpectralClusteringPaths()
	
	    def generate_path_groups(self, num_clusters: int = None) -> None:
	        path_group = {}
	        for k, v in self.memory_kernel_dict.items():
	            A_k = v.A_k
	            if num_clusters is None:
	                num_clusters = int(np.sqrt(len(A_k)))
	            paths = self.cluster_paths.create_paths(A_k, num_clusters)
	            path_group[k] = paths
	        self.path_group = path_group
	

.. automodule:: multi_kernel
   :members:
