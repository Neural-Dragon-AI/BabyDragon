from abc import ABC, abstractmethod
from typing import List, Optional, Union, Tuple, Any
from babydragon.models.embedders.ada2 import OpenAiEmbedder
from babydragon.models.embedders.cohere import CohereEmbedder
import numpy as np
import os
import json
import collections

class BaseIndex(ABC):
    def __init__(
            self,
            values: Optional[List[str]] = None,
            embeddings: Optional[List[Union[List[float], np.ndarray]]] = None,
            name: str = "np_index",
            save_path: Optional[str] = None,
            load: bool = False,
            embedder: Optional[Union[OpenAiEmbedder, CohereEmbedder]] = OpenAiEmbedder,
    ):
        self.name = name
        self.embedder = embedder()
        self.save_path = save_path or "storage"
        os.makedirs(self.save_path, exist_ok=True)
        self.values = []
        self.embeddings = None  # initialize embeddings as None
        self.index_set = set()  # add this to quickly check for duplicates
        self.loaded = False
        self.setup_index(values, embeddings, load)
    
    @staticmethod
    @abstractmethod
    def compare_embeddings(query: Any, targets: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def batched_l2_distance(query_embedding: Any, embeddings: Any, mask: Optional[Any] = None) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def batched_cosine_similarity(query_embedding: Any, embeddings: Any, mask: Optional[Any] = None) -> Any:
        pass

    @abstractmethod
    def get(self, identifier: Union[int, str, np.ndarray, List[Union[int, str, np.ndarray]]]) -> Union[str, List[str]]:
        pass

    @abstractmethod
    def add(self, values: List[str], embeddings: Optional[List[Union[List[float], np.ndarray]]] = None):
        pass

    @abstractmethod
    def remove(self, identifier: Union[int, str, np.ndarray, List[Union[int, str, np.ndarray]]]) -> None:
        pass

    @abstractmethod
    def update(self, old_identifier: Union[int, str, np.ndarray, List[Union[int, str, np.ndarray]]], new_value: Union[str, List[str]], new_embedding: Optional[Union[List[float], np.ndarray, List[List[float]], List[np.ndarray]]] = None) -> None:
        pass

    @abstractmethod
    def search(self, query: Optional[str] = None, query_embedding: Optional[np.ndarray] = None, top_k: int = 10, metric: str = "cosine", filter_mask: Optional[np.ndarray] = None) -> Tuple[List[str], Optional[List[float]], List[int]]:
        pass

    # Non-abstract method
    def save_index(self):
        save_directory = os.path.join(self.save_path, self.name)
        os.makedirs(save_directory, exist_ok=True)

        with open(os.path.join(save_directory, f"{self.name}_values.json"), "w") as f:
            json.dump(self.values, f)

        # Save embeddings in a subclass-specific way
        self._save_embeddings(save_directory)

    @abstractmethod
    def _save_embeddings(self, directory: str):
        pass
