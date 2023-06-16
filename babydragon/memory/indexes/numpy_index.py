from typing import List, Optional, Union, Tuple
from babydragon.models.embedders.ada2 import OpenAiEmbedder
from babydragon.models.embedders.cohere import CohereEmbedder
import numpy as np
import os
import json
import collections
from babydragon.memory.indexes.base_index import BaseIndex

class NpIndex(BaseIndex):
    def __init__(
            self,
            values: Optional[List[str]] = None,
            embeddings: Optional[List[Union[List[float], np.ndarray]]] = None,
            name: str = "np_index",
            save_path: Optional[str] = None,
            load: bool = False,
            embedder: Optional[Union[OpenAiEmbedder, CohereEmbedder]] = OpenAiEmbedder,
    ):
        BaseIndex.__init__(self,values, embeddings, name, save_path, load, embedder)

    @staticmethod
    def compare_embeddings(query: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return np.all(query == targets, axis=1)

    @staticmethod
    def batched_l2_distance(query_embedding: np.ndarray, embeddings: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        scores = np.linalg.norm(embeddings - query_embedding, axis=1)
        if mask is not None:
            scores[~mask.astype(bool)] = np.inf  # set scores of excluded embeddings to infinity
        return scores

    @staticmethod
    def batched_cosine_similarity(query_embedding: np.ndarray, embeddings: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        scores = np.dot(embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True),
                        query_embedding / np.linalg.norm(query_embedding))
        if mask is not None:
            scores[~mask.astype(bool)] = -np.inf  # set scores of excluded embeddings to negative infinity
        return scores

    def _save_embeddings(self, directory: str):
        np.save(os.path.join(directory, f"{self.name}_embeddings.npy"), self.embeddings)


    def load_index(self):
        load_directory = os.path.join(self.save_path, self.name)
        if not os.path.exists(load_directory):
            print(f"I did not find the directory to load the index from: {load_directory}")
            return

        print(f"Loading index from {load_directory}")

        with open(os.path.join(load_directory, f"{self.name}_values.json"), "r") as f:
            self.values = json.load(f)

        self.embeddings = np.load(os.path.join(load_directory, f"{self.name}_embeddings.npy"))
        print(self.embeddings.shape, len(self.values))
        if len(self.values) != len(self.embeddings):
            raise ValueError("Loaded values and embeddings are not the same length.")

        self.loaded = True

    def setup_index(self, input_values: Optional[List[str]], embeddings: Optional[List[Union[List[float], np.ndarray]]], load: bool):
        if load and os.path.exists(os.path.join(self.save_path, self.name)):
            self.load_index()
                 
        elif input_values and embeddings and len(input_values) == len(embeddings):
            # Check that input_values and embeddings are the same length
            unique_dict = collections.defaultdict(list)
            for val, emb in zip(input_values, embeddings):
                unique_dict[val].append(emb)

            # Ensure that all embeddings for each value are identical
            for val in unique_dict.keys():
                if len(unique_dict[val]) > 1 and not all(np.array_equal(unique_dict[val][0], emb) for emb in unique_dict[val]):
                    raise ValueError(f'Different embeddings for the same value "{val}" found.')

            self.add(list(unique_dict.keys()), [unique_dict[val][0] for val in unique_dict.keys()])
            self.save_index()

        elif input_values:
            # Embed the input_values
            self.add(list(set(input_values)))
            self.save_index()


    def add(self, values: List[str], embeddings: Optional[List[Union[List[float], np.ndarray]]] = None):
        if embeddings is None:
            embeddings = self.embedder.embed(values)
        elif len(values) != len(embeddings):
            raise ValueError("values and embeddings must be the same length")

        # Check for duplicates and only add unique values
        unique_values = [value for value in values if value not in self.index_set]
        unique_embeddings = [embedding for value, embedding in zip(values, embeddings) if value not in self.index_set]
        if not unique_values:
            print("All values already exist in the index. No values were added.")
            return

        # Add unique values to the set
        self.index_set.update(unique_values)
        
        # If embeddings array is not yet created, initialize it, else append to it
        if self.embeddings is None:
            self.embeddings = np.array(unique_embeddings)
        else:
            self.embeddings = np.vstack((self.embeddings, unique_embeddings))
        
        self.values.extend(unique_values)

    def remove(self, identifier: Union[int, str, np.ndarray, List[Union[int, str, np.ndarray]]]) -> None:
        if isinstance(identifier, list):
            if all(isinstance(i, type(identifier[0])) for i in identifier):
                for i in identifier:
                    self.remove(i)
            else:
                raise TypeError("All elements in the list must be of the same type.")
        elif isinstance(identifier, int):  # if given an index
            if identifier < len(self.values):  # if valid index
                self.index_set.remove(self.values[identifier])
                self.values = [v for i, v in enumerate(self.values) if i != identifier]
                self.embeddings = self.embeddings[np.arange(len(self.embeddings))!=identifier]
            else:
                raise ValueError("Invalid index given for removal.")
        elif isinstance(identifier, str):  # if given a value
            if identifier in self.values:
                index = self.values.index(identifier)
                self.index_set.remove(identifier)
                self.values = [v for i, v in enumerate(self.values) if i != index]
                self.embeddings = self.embeddings[np.arange(len(self.embeddings))!=index]
            else:
                raise ValueError("Value not found for removal.")
        elif isinstance(identifier, np.ndarray):  # if given an embedding
            # Find indices of embeddings that are equal to the given one
            indices = np.where(self.compare_embeddings(identifier, self.embeddings))[0]
            if len(indices) == 0:
                raise ValueError("Embedding not found for removal.")
            for i in reversed(indices):
                self.index_set.remove(self.values[i])
                self.values.pop(i)
            self.embeddings = np.delete(self.embeddings, indices, axis=0)
        else:
            raise TypeError("Invalid identifier type. Expected int, str, np.ndarray, or list of these types")
        
    def update(self, old_identifier: Union[int, str, np.ndarray, List[Union[int, str, np.ndarray]]], new_value: Union[str, List[str]], new_embedding: Optional[Union[List[float], np.ndarray, List[List[float]], List[np.ndarray]]] = None) -> None:
        if isinstance(old_identifier, list):
            if not isinstance(new_value, list) or len(old_identifier) != len(new_value):
                raise ValueError("For list inputs, old_identifier and new_value must all be lists of the same length.")
            if new_embedding is not None:
                if not isinstance(new_embedding, list) or len(old_identifier) != len(new_embedding):
                    raise ValueError("new_embedding must be a list of same length as old_identifier and new_value.")
                # If new_embedding is a 2D array or list of lists
                if isinstance(new_embedding[0], list) or isinstance(new_embedding[0], np.ndarray):
                    if len(new_embedding[0]) != len(self.embeddings[0]):
                        raise ValueError("Each row in new_embedding must have the same dimension as the original embeddings.")
            else:
                new_embedding = self.embedder.embed(new_value)
            for old_id, new_val, new_emb in zip(old_identifier, new_value, new_embedding):
                self.update(old_id, new_val, new_emb)
        elif isinstance(old_identifier, int):  # if given an index
            if old_identifier < len(self.values):  # if valid index
                self.index_set.remove(self.values[old_identifier])
                self.values[old_identifier] = new_value
                self.index_set.add(new_value)
                self.embeddings[old_identifier] = self.embedder.embed([new_value])[0] if new_embedding is None else new_embedding
            else:
                raise ValueError("Invalid index given for update.")
        elif isinstance(old_identifier, str):  # if given a value
            if old_identifier in self.values:
                index = self.values.index(old_identifier)
                self.index_set.remove(old_identifier)
                self.values[index] = new_value
                self.index_set.add(new_value)
                self.embeddings[index] = self.embedder.embed([new_value])[0] if new_embedding is None else new_embedding
            else:
                raise ValueError("Value not found for update.")
        elif isinstance(old_identifier, np.ndarray):  # if given an embedding
            # Find indices of embeddings that are equal to the given one
            indices = np.where(self.compare_embeddings(old_identifier, self.embeddings))[0]
            if len(indices) == 0:
                raise ValueError("Embedding not found for update.")
            for i in indices:
                self.index_set.remove(self.values[i])
                self.values[i] = new_value
                self.index_set.add(new_value)
                self.embeddings[i] = self.embedder.embed([new_value])[0] if new_embedding is None else new_embedding
        else:
            raise TypeError("Invalid identifier type. Expected int, str, np.ndarray, or list of these types")

    def search(self, query: Optional[str] = None, query_embedding: Optional[np.ndarray] = None, top_k: int = 10, metric: str = "cosine", filter_mask: Optional[np.ndarray] = None) -> List[Tuple[int, float]]:
        # create a 2D numpy array from the embeddings list
        embeddings_array = self.embeddings

        # if no query or query embedding is provided, return random samples
        if query is None and query_embedding is None:
            indices = np.random.choice(len(self.values), size=top_k, replace=False)
            return [self.values[i] for i in indices],None, indices  # return indices with dummy scores

        # initialize a cache for storing unique queries and their embeddings
        if not hasattr(self, 'query_cache'):
            self.query_cache = {}

        # if the query is in the cache, use the stored embedding
        if query in self.query_cache:
            print("Query is in cache.")
            query_embedding = self.query_cache[query]
        # else if a query string is provided but not in cache, compute its embedding
        elif query is not None:
            print("Query is not in cache. Computing embedding...")
            query_embedding = self.embedder.embed([query])
            print(query_embedding)
            self.query_cache[query] = query_embedding  # store the new query and its embedding in cache

        # compute distances or similarities
        if metric == "l2":
            scores = self.batched_l2_distance(query_embedding, embeddings_array, filter_mask)
        elif metric == "cosine":
            scores = self.batched_cosine_similarity(query_embedding, embeddings_array, filter_mask)
        else:
            raise ValueError("Invalid metric. Expected 'l2' or 'cosine'.")

        # sort by scores
        sorted_indices = np.argsort(scores)

        # for L2 distance, closer vectors are better (smaller distance)
        # for cosine similarity, further vectors are better (larger similarity)
        top_k = min(top_k, len(self.values))
        top_indices = sorted_indices[:top_k] if metric == "l2" else sorted_indices[-top_k:][::-1]

        # return indices and scores
        return  [self.values[i] for i in top_indices],[scores[i] for i in top_indices], [i for i in top_indices]
