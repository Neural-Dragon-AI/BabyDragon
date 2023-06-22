from typing import List, Optional, Union, Tuple, Dict
from babydragon.models.embedders.ada2 import OpenAiEmbedder, TOKENIZER, MAX_CONTEXT_LENGTH
from babydragon.models.embedders.cohere import CohereEmbedder
import numpy as np
import os
import collections
from babydragon.memory.indexes.base_index import BaseIndex
from babydragon.utils.main_logger import logger


class NpIndex(BaseIndex):
    def __init__(
            self,
            values: Optional[List[str]] = None,
            embeddings: Optional[List[Union[List[float], np.ndarray]]] = None,
            name: str = "np_index",
            save_path: Optional[str] = None,
            load: bool = False,
            embedder: Optional[Union[OpenAiEmbedder, CohereEmbedder]] = OpenAiEmbedder,
            token_overflow_strategy: str = "ignore",
    ):
        self.old_ids = collections.OrderedDict()
        BaseIndex.__init__(self,values=values, embeddings=embeddings, name=name, save_path=save_path, load=load, embedder=embedder, token_overflow_strategy=token_overflow_strategy)


    @staticmethod
    def compare_embeddings(query: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return np.array([np.allclose(query, target, rtol=1e-05, atol=1e-08) for target in targets])

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
        if self.queries_embeddings is not None:
            np.save(os.path.join(directory, f"{self.name}_queries_embeddings.npy"), self.queries_embeddings)


    def _load_embeddings(self, directory: str):
        load_directory = os.path.join(self.save_path, self.name)
        if not os.path.exists(load_directory):
            print(f"I did not find the directory to load the embe from: {load_directory}")
            return

        self.embeddings = np.load(os.path.join(load_directory, f"{self.name}_embeddings.npy"))
        if len(self.values) != len(self.embeddings):
            raise ValueError("Loaded values and embeddings are not the same length.")
        #check that queries embeddings exist
        if os.path.exists(os.path.join(load_directory, f"{self.name}_queries_embeddings.npy")):
            self.queries_embeddings = np.load(os.path.join(load_directory, f"{self.name}_queries_embeddings.npy"),allow_pickle=True)
            print(self.embeddings.shape, len(self.values))
            print(self.queries_embeddings.shape, len(self.queries))
            print(self.queries, self.queries_embeddings, self.queries_embeddings is not None, type(self.queries_embeddings), self.queries_embeddings.shape)

            if self.queries_embeddings is not None and len(self.queries) != len(self.queries_embeddings):
                raise ValueError("Loaded queries and queries embeddings are not the same length.")


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

    def get(self, identifier: Union[int, str, np.ndarray, List[Union[int, str, np.ndarray]]], output_type: str = "value") -> Union[int, str, np.ndarray, Dict[str, Union[int, str, np.ndarray]]]:

        index = self.identify_input(identifier)
        # Define output types
        output_types = {
            'index': index,
            'value': self.values[index],
            'embedding': self.embeddings[index],
            'all': {
                'index': index,
                'value': self.values[index],
                'embedding': self.embeddings[index]}
        }

        # Check output type is valid
        if output_type not in output_types:
            raise ValueError("Invalid output_type. Expected 'index', 'value', or 'embedding'.")

        return output_types[output_type]

    def validate_value_length(self, value: str, tokens: List[int]) -> Union[bool, str]:
        if not isinstance(value, str):
            raise TypeError("Value must be a string.")
        token_len = len(tokens)
        overflow_check = token_len > MAX_CONTEXT_LENGTH
        match self.token_overflow_strategy:
            case "ignore":
                if overflow_check:
                    return False, value
                else:
                    return True, value
            case "truncate":
                if overflow_check:
                    return True, TOKENIZER.decode(tokens[:MAX_CONTEXT_LENGTH])
                else:
                    return True, value
            case "error":
                if overflow_check:
                    raise ValueError(f" The input is too long for OpenAI, num tokens is {token_len}, instead of {MAX_CONTEXT_LENGTH}")
                else:
                    return True, value
            case _:
                raise ValueError("Invalid token_overflow_strategy. Expected 'ignore', 'truncate', or 'error'.")

    def add(self, values: List[str], embeddings: Optional[List[Union[List[float], np.ndarray]]] = None):

        if embeddings is not None and len(values) != len(embeddings):
            raise ValueError("values and embeddings must be the same length")

        # Check for duplicates and only add unique values
        unique_values = []
        unique_embeddings = []
        token_batch = TOKENIZER.encode_batch(values, allowed_special="all")
        #extract the max value in old_ids consider that each old_ids is a list take the max of the list itself
        for i, (val, tokens) in enumerate(zip(values, token_batch)):
            is_valid, value = self.validate_value_length(val, tokens)
            if not is_valid:
                print(f"Value '{value}' is too long and will be ignored.")
                continue
            if value not in self.index_set:
                unique_values.append(value)
                self.index_set.add(value)

                self.old_ids[value] = [i]
                if embeddings is not None:
                    unique_embeddings.append(embeddings[i])
            else:
                self.old_ids[value].append(i)

        if not unique_values:
            print("All values already exist in the index. No values were added.")
            return
        if embeddings is None:
            unique_embeddings = self.embedder.embed(unique_values)

        # Add unique values to the set
        self.index_set.update(unique_values)

        # If embeddings array is not yet created, initialize it, else append to it
        if self.embeddings is None:
            logger.info("Initializing embeddings array")
            self.embeddings = np.vstack(unique_embeddings)
        else:
            self.embeddings = np.vstack((self.embeddings, unique_embeddings))

        self.values.extend(unique_values)

    def identify_input(self, identifier: Union[int, str, np.ndarray, List[Union[int, str, np.ndarray]]]) -> Union[int, str, np.ndarray]:
        if isinstance(identifier, int):  # if given an index
            if identifier < len(self.values):  # if valid index
                index = identifier
            else:
                raise ValueError("Invalid index given.")
        elif isinstance(identifier, str):  # if given a value
            if identifier in self.values:
                index = self.values.index(identifier)
            else:
                raise ValueError("Value not found.")
        elif isinstance(identifier, np.ndarray):  # if given an embedding
            indices = np.where(self.compare_embeddings(identifier, self.embeddings))[0]
            if len(indices) == 0:
                raise ValueError("Embedding not found.")
            index = indices[0]
        else:
            raise TypeError("Invalid identifier type. Expected int, str, np.ndarray, or list of these types")

        return index


    def remove(self, identifier: Union[int, str, np.ndarray, List[Union[str, np.ndarray]]]) -> None:

        if isinstance(identifier, list):
            if all(isinstance(i, type(identifier[0])) for i in identifier) and not isinstance(identifier[0], list) and not isinstance(identifier[0], int):
                for i in identifier:
                    self.remove(i)
            else:
                raise TypeError("All elements in the list must be of the same type.")
            return
        id = self.identify_input(identifier)
        value = self.values[id]
        self.index_set.remove(value)
        self.old_ids.pop(value)
        self.values.pop(id)
        self.embeddings = np.delete(self.embeddings, [id], axis=0)


    def update(self, old_identifier: Union[int, str, np.ndarray, List[Union[str, np.ndarray]]], new_value: Union[str, List[str]], new_embedding: Optional[Union[List[float], np.ndarray, List[List[float]], List[np.ndarray]]] = None) -> None:
        if isinstance(new_value,str) and new_value in self.index_set:
            raise ValueError("new_value already exists in the index. Please remove it first.")
        elif isinstance(new_value, list) and any(v in self.index_set for v in new_value):
            raise ValueError("One or more new_value already exists in the index. Please remove them first.")
        if isinstance(old_identifier, list) and not isinstance(old_identifier[0], list) and not isinstance(old_identifier[0], int):
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
            return
        old_id = self.identify_input(old_identifier)
        self.index_set.remove(self.values[old_id])
        self.old_ids[new_value] = self.old_ids.pop(self.values[old_id])
        self.values[old_id] = new_value
        self.index_set.add(new_value)
        self.embeddings[old_id] = self.embedder.embed([new_value]) if new_embedding is None else new_embedding

    def search(self, query: Optional[str] = None, query_embedding: Optional[np.ndarray] = None, top_k: int = 10, metric: str = "cosine", filter_mask: Optional[np.ndarray] = None) -> Tuple[List[str], Optional[List[float]], List[int]]:

        # create a 2D numpy array from the embeddings list
        embeddings_array = self.embeddings

        # if no query or query embedding is provided, return random samples
        if query is None and query_embedding is None:
            indices = np.random.choice(len(self.values), size=top_k, replace=False)
            return [self.values[i] for i in indices],None, indices  # return indices with dummy scores

        # if the query is in the queries set, use the stored embedding
        if query in self.queries_set:
            print("Query is in queries set.")
            query_embedding = self.queries_embeddings[self.queries.index(query)]
        # else if a query string is provided but not in queries set, compute its embedding
        elif query is not None:
            print("Query is not in queries set. Computing embedding...")
            query_embedding = self.embedder.embed([query])
            # print(query_embedding)
            self.queries_set.add(query)
            self.queries.append(query)
            # If queries_embeddings array is not yet created, initialize it, else append to it
            if self.queries_embeddings is None:
                self.queries_embeddings = np.array([query_embedding])
            else:
                self.queries_embeddings = np.vstack((self.queries_embeddings, query_embedding))

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
