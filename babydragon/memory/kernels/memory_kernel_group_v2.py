from babydragon.memory.kernels.memory_kernel import MemoryKernel
from babydragon.chat.chat import Chat
from typing import Dict
from babydragon.tasks.llm_task import LLMWriter
from babydragon.memory.kernels.kernel_clustering import HDBSCANPaths, SpectralClusteringPaths
import numpy as np

class MemoryKernelGroup(MemoryKernel):
    def __init__(self, memory_kernel_dict: Dict[str, MemoryKernel], name="memory_kernel_group"):
        """
        Initialize the MemoryKernelGroup with a dictionary of MemoryKernel instances.

        Args:
            memory_kernel_dict (Dict[str, MemoryKernel]): A dictionary of MemoryKernel instances.
            name (str, optional): The name of the MemoryKernelGroup. Defaults to "memory_kernel_group".
        """
        self.memory_kernel_dict = memory_kernel_dict
        self.path_group = {}
        self.name = name

    def gen_index_aligned_kernel(
        self, chatbot: Chat, parent_kernel_label: str, child_kernel_label: str
    ) -> None:
        """
        Generate an index-aligned kernel using LLMWriter for the given parent and child kernel labels.

        Args:
            chatbot (Chat): The Chat instance.
            parent_kernel_label (str): The label of the parent kernel.
            child_kernel_label (str): The label of the child kernel.
        """
        llm_writer = LLMWriter(
            index=self.memory_kernel_dict[parent_kernel_label],
            path=self.path_group[parent_kernel_label],
            chatbot=chatbot,
            write_func=None,
            max_workers=1,
        )
        new_index = llm_writer.write()
        new_memory_kernel = MemoryKernel(mem_index=new_index, name=child_kernel_label)
        new_memory_kernel.create_k_hop_index(k=2)
        self.memory_kernel_dict[child_kernel_label] = new_memory_kernel

class HDBSCANMemoryKernelGroup(MemoryKernelGroup):
    def __init__(self, memory_kernel_dict: Dict[str, MemoryKernel], name="memory_kernel_group"):
        super().__init__(memory_kernel_dict, name)
        self.cluster_paths = HDBSCANPaths()

    def generate_path_groups(self) -> None:
        path_group = {}
        for k, v in self.memory_kernel_dict.items():
            embeddings = v.node_embeddings
            num_clusters = int(np.sqrt(len(embeddings)))
            paths = self.cluster_paths.create_paths(embeddings, num_clusters)
            path_group[k] = paths
        self.path_group = path_group

class SpectralClusteringMemoryKernelGroup(MemoryKernelGroup):
    def __init__(self, memory_kernel_dict: Dict[str, MemoryKernel], name="memory_kernel_group"):
        super().__init__(memory_kernel_dict, name)
        self.cluster_paths = SpectralClusteringPaths()

    def generate_path_groups(self) -> None:
        path_group = {}
        for k, v in self.memory_kernel_dict.items():
            embeddings = v.node_embeddings
            num_clusters = int(np.sqrt(len(embeddings)))
            paths = self.cluster_paths.create_paths(embeddings, num_clusters)
            path_group[k] = paths
        self.path_group = path_group
