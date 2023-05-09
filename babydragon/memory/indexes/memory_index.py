import copy
import io
import json
import os
import pickle
import re
import time
from typing import Dict, List, Optional, Tuple, Union

import faiss
import numpy as np
import pandas as pd
import tiktoken
from IPython.display import Markdown, display

from babydragon.models.embedders.ada2 import OpenAiEmbedder
from babydragon.models.embedders.cohere import CohereEmbedder
from babydragon.tasks.embedding_task import parallel_embeddings
from datasets import load_dataset
from babydragon.utils.hf_datasets import extract_values_and_embeddings_hf, extract_values_hf
from babydragon.utils.pandas import extract_values_and_embeddings

from typing import List, Optional, Tuple
import re
import numpy as np

def prune_index(
    cls: "MemoryIndex",
    constraint: Optional[str] = None,
    regex_pattern: Optional[str] = None,
    length_constraint: Optional[Tuple[int,int]] = None,
    tokenizer: Optional[tiktoken.Encoding] = None,
) -> "MemoryIndex" :
    if constraint is not None:
        if constraint == "regex":
            if regex_pattern is None:
                raise ValueError("regex_pattern must be provided for regex constraint.")
            pruned_values, pruned_embeddings = _prune_by_regex(cls, regex_pattern)
        elif constraint == "length":
            if length_constraint is None:
                raise ValueError("length_constraint must be provided for length constraint.")
            pruned_values, pruned_embeddings = _prune_by_length(cls, length_constraint, tokenizer)
        else:
            raise ValueError("Invalid constraint type provided.")
    else:
        raise ValueError("constraint must be provided for pruning the index.")

    pruned_memory_index = cls.__class__(
        values=pruned_values,
        embeddings=pruned_embeddings,
        name=cls.name + "_pruned",
    )

    return pruned_memory_index


def _prune_by_regex(cls: "MemoryIndex", regex_pattern: str) -> Tuple[List[str], List[np.ndarray]]:
    pruned_values = []
    pruned_embeddings = []

    for value in cls.values:
        if re.search(regex_pattern, value):
            pruned_values.append(value)
            pruned_embeddings.append(cls.get_embedding_by_value(value))

    return pruned_values, pruned_embeddings


def _prune_by_length(cls: "MemoryIndex", length_constraint: Tuple[int,int], tokenizer) -> Tuple[List[str], List[np.ndarray]]:
    pruned_values = []
    pruned_embeddings = []
    if tokenizer is None:
        len_func = len
    else:
        def len_func(value):
            return len(tokenizer.encode(value))
    print("Pruning by length")
    print("Length constraint: ", length_constraint)
    print("Number of values: ", len(cls.values))
    print("tokenizer: ", tokenizer)
    for value in cls.values:
        if len_func(value) <= length_constraint[1] and len_func(value) >= length_constraint[0]:
            print(f"value {value} is in range {length_constraint}")
            pruned_values.append(value)
            pruned_embeddings.append(cls.get_embedding_by_value(value))

    return pruned_values, pruned_embeddings

def save(cls):
    save_directory = os.path.join(cls.save_path, cls.name)
    os.makedirs(save_directory, exist_ok=True)

    index_filename = os.path.join(save_directory, f"{cls.name}_index.faiss")
    faiss.write_index(cls.index, index_filename)

    values_filename = os.path.join(save_directory, f"{cls.name}_values.json")
    with open(values_filename, "w") as f:
        json.dump(cls.values, f)

    embeddings_filename = os.path.join(save_directory, f"{cls.name}_embeddings.npz")
    np.savez_compressed(embeddings_filename, cls.get_all_embeddings())

def load(cls):
    load_directory = os.path.join(cls.save_path, cls.name)
    if not os.path.exists(load_directory):
        cls.loaded = False
        print("I did not find the directory to load the index from.", load_directory)
        return

    print(f"Loading index from {load_directory}")

    index_filename = os.path.join(load_directory, f"{cls.name}_index.faiss")
    cls.index = faiss.read_index(index_filename)

    values_filename = os.path.join(load_directory, f"{cls.name}_values.json")
    with open(values_filename, "r") as f:
        cls.values = json.load(f)

    embeddings_filename = os.path.join(load_directory, f"{cls.name}_embeddings.npz")
    embeddings_data = np.load(embeddings_filename)
    cls.embeddings = embeddings_data["arr_0"]
    cls.loaded = True

class MemoryIndex:
    """
    this class is a wrapper for a faiss index, it contains information about the format of the index the faiss index itself
    """

    def __init__(
        self,
        index: Optional[faiss.Index] = None,
        values: Optional[List[str]] = None,
        embeddings: Optional[List[Union[List[float], np.ndarray]]] = None,
        name: str = "memory_index",
        save_path: Optional[str] = None,
        load: bool = False,
        tokenizer: Optional[tiktoken.Encoding] = None,
        max_workers: int = 1,
        backup: bool = False,
        embedder: Optional[Union[OpenAiEmbedder,CohereEmbedder]] = OpenAiEmbedder,
        is_batched: bool = False,
    ):

        self.name = name
        self.embedder = embedder()
        self.save_path = save_path if save_path is not None else "storage"
        os.makedirs(self.save_path, exist_ok=True)
        self.values = []
        self.embeddings = []
        self.max_workers = max_workers
        self.is_batched = is_batched

        if load is True:
            self.load()
        else:
            self.loaded = False
        if not self.loaded:
            if (
                self.max_workers > 1
                and values is not None
                and embeddings is None
                and index is None
            ):
                embeddings = parallel_embeddings(self.embedder,
                    values, max_workers, backup=backup, name=name
                )
            self.init_index(index, values, embeddings, is_embed_batched=is_batched)
        if tokenizer is None:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        else:
            self.tokenizer = tokenizer
        self.query_history = []
        self.save()

    def init_index(
        self,
        index: Optional[faiss.Index] = None,
        values: Optional[List[str]] = None,
        embeddings: Optional[List[Union[List[float], np.ndarray]]] = None,
        is_embed_batched: bool = False,
    ) -> None:

        """
        initializes the index, there are 4 cases:
        1. we create a new index from scratch
        2. we create a new index from a list of embeddings and values
        3. we create a new index from a faiss index and values list
        4. we load an index from a file
        """
        if index is None and values is None and embeddings is None:
            print("Creating a new index")
            self.index = faiss.IndexFlatIP(self.embedder.get_embedding_size())
            self.values = []
        elif (
            index is None
            and values is not None
            and embeddings is not None
            and len(values) == len(embeddings)
        ):
            print("Creating a new index from a list of embeddings and values")
            self.index = faiss.IndexFlatIP(self.embedder.get_embedding_size())
            #add all the embeddings to the index
            if is_embed_batched:
                print("Adding batched embeddings to index")
                print(type(embeddings))
                embeddings = np.array(embeddings)
                self.add_batch_to_index(values=values, embeddings=embeddings)
            else:
                for embedding, value in zip(embeddings, values):
                    self.add_to_index(value, embedding)

        elif (
            isinstance(index, faiss.Index)
            and index.d == self.embedder.get_embedding_size()
            and type(values) == list
            and len(values) == index.ntotal
        ):
            print("Creating a new index from a faiss index and values list")
            self.index = index
            self.values = values
        elif index is None and values is not None and embeddings is None:
            print("Creating a new index from a list of values")
            self.index = faiss.IndexFlatIP(self.embedder.get_embedding_size())
            if is_embed_batched:
                batch = []
                i = 0
                for value in values:
                    batch.append(value)
                    if len(batch) == 1000:
                        start = time.time()
                        self.add_batch_to_index(values=batch)
                        print(f"Embedding batch {i} took ", time.time() - start, " seconds")
                        print(f"Batch {i} of {len(values)//1000}")
                        i +=1
                        batch = []
                if len(batch) > 0:
                    self.add_batch_to_index(values=batch)
            else:
                i = 0
                for value in values:
                    print("Embedding value ", i, " of ", len(values))
                    start = time.time()
                    self.add_to_index(value)
                    print("Embedding value ", i, " took ", time.time() - start, " seconds")
                    i += 1
        else:
            print(type(values))
            print(type(embeddings))
            print(type(index))
            raise ValueError(
                "The index is not a valid faiss index or the embedding dimension is not correct"
            )
        print(len(self.values), " values in the index")
        print(self.index.ntotal, " embeddings in the index")

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["index"]

        index_buffer = io.BytesIO()
        faiss.write_index(state["index"], index_buffer)
        state["index_bytes"] = index_buffer.getvalue()

        return state

    def __setstate__(self, state):
        index_buffer = io.BytesIO(state["index_bytes"])
        state["index"] = faiss.read_index(index_buffer)

        del state["index_bytes"]

        self.__dict__.update(state)

    @classmethod
    def from_pandas(
        cls,
        data_frame: Union[pd.DataFrame, str],
        columns: Optional[Union[str, List[str]]] = None,
        name: str = "memory_index",
        save_path: Optional[str] = None,
        in_place: bool = True,
        embeddings_col: Optional[str] = None,
    ) -> "MemoryIndex":
        """
        Initialize a MemoryIndex object from a pandas DataFrame.

        Args:
            data_frame: The DataFrame or path to a CSV file.
            columns: The columns of the DataFrame to use as values.
            name: The name of the index.
            save_path: The path to save the index.
            in_place: Whether to work on the DataFrame in place or create a copy.
            embeddings_col: The column name containing the embeddings.

        Returns:
            A MemoryIndex object initialized with values and embeddings from the DataFrame.
        """

        if (
            isinstance(data_frame, str)
            and data_frame.endswith(".csv")
            and os.path.isfile(data_frame)
        ):
            print("Loading the CSV file")
            try:
                data_frame = pd.read_csv(data_frame)
            except:
                raise ValueError("The CSV file is not valid")
            name = data_frame.split("/")[-1].split(".")[0]
        elif isinstance(data_frame, pd.core.frame.DataFrame) and columns is not None:
            print("Loading the DataFrame")
            if not in_place:
                data_frame = copy.deepcopy(data_frame)
        else:
            raise ValueError(
                "The data_frame is not a valid pandas dataframe or the columns are not valid or the path is not valid"
            )

        values, embeddings = extract_values_and_embeddings(
            data_frame, columns, embeddings_col
        )
        return cls(values=values, embeddings=embeddings, name=name, save_path=save_path)

    @classmethod
    def from_hf_dataset(
        cls,
        dataset_url: str,
        value_column: str,
        embeddings_column: Optional[str] = None,
        name: str = "memory_index",
        save_path: Optional[str] = None,
        embeddings_type: Optional[Union[OpenAiEmbedder,CohereEmbedder]]= CohereEmbedder,
        is_batched: bool = False,
    ) -> "MemoryIndex":
        """
        Initialize a MemoryIndex object from a Hugging Face dataset.

        Args:
            dataset_url: The URL of the Hugging Face dataset.
            value_column: The column of the dataset to use as values.
            embeddings_column: The column of the dataset containing the embeddings.
            name: The name of the index.
            save_path: The path to save the index.

        Returns:
            A MemoryIndex object initialized with values and embeddings from the Hugging Face dataset.
        """
        dataset = load_dataset(dataset_url)['train']
        if embeddings_column is not None:
            values, embeddings = extract_values_and_embeddings_hf(
                dataset, value_column, embeddings_column
            )
        elif embeddings_column is None:
            values = extract_values_hf(dataset, value_column)
            embeddings = None
        else:
            raise ValueError(
                "The dataset is not a valid Hugging Face dataset or the columns are not valid"
            )
        return cls(values=values, embeddings=embeddings, name=name, save_path=save_path, embedder=embeddings_type, is_batched=is_batched)

    def add_to_index(
        self,
        value: str,
        embedding: Optional[Union[List[float], np.ndarray, str]] = None,
        verbose: bool = False,
        default_save: bool = False,
    ) -> None:
        """
        index a message in the faiss index, the message is embedded (if embedding is not provided) and the id is saved in the values list
        """
        if value not in self.values:
            if embedding is None:
                embedding = self.embedder.embed(value)
                if verbose:
                    display(
                        Markdown("The value {value} was embedded".format(value=value))
                    )
                if type(embedding) is list:
                    embedding = np.array([embedding])
                self.index.add(embedding)
                self.values.append(value)
            elif embedding is not None:
                if type(embedding) is list:
                    embedding = np.array([embedding])
                elif type(embedding) is str:
                    try:
                        embedding = eval(embedding)
                        embedding = np.array([embedding]).astype(np.float32)
                    except (SyntaxError, ValueError):
                        print("The string is not a valid list, probably an error:", embedding)
                        return
                elif type(embedding) is not np.ndarray:
                    raise ValueError("The embedding is not a valid type")

                if embedding.ndim == 1:
                    embedding = embedding.reshape(1, -1)
                self.index.add(embedding)
                self.values.append(value)
                if default_save:
                    self.save()  # we should check here the save time is not too long
        else:
            if verbose:
                display(
                    Markdown(
                        "The value {value} was already in the index".format(value=value)
                    )
                )
    def add_batch_to_index(
        self,
        values: List[str],
        embeddings: Optional[Union[List[float], np.ndarray, str]] = None,
        verbose: bool = False,
        default_save: bool = False,
    ) -> None:
        """
        index a message in the faiss index, the message is embedded (if embedding is not provided) and the id is saved in the values list
        """

        if embeddings is None:
            embeddings = self.embedder.batch_embed(values)
            if verbose:
                display(
                    Markdown("The value batch was embedded")
                )
            if type(embeddings) is list:
                embeddings = np.array(embeddings)
            self.index.add(embeddings)
            self.values.extend(values)
        elif embeddings is not None:
            if type(embeddings) is list:
                embeddings = np.array([embeddings])
            elif type(embeddings) is str:
                try:
                    embeddings = eval(embeddings)
                    embeddings = np.array([embeddings]).astype(np.float32)
                except (SyntaxError, ValueError):
                    print("The string is not a valid list, probably an error:", embeddings)
                    return
            elif type(embeddings) is not np.ndarray:
                raise ValueError("The embedding is not a valid type")

            self.index.add(embeddings)
            self.values.extend(values)
            if default_save:
                self.save()  # we should check here the save time is not too long


    def remove_from_index(self, value: str) -> None:
        """
        Remove a value from the index and the values list.
        Args:
            value: The value to remove from the index.
        """
        index = self.get_index_by_value(value)
        if index is not None:
            self.values.pop(index)

            id_selector = faiss.IDSelectorArray(np.array([index], dtype=np.int64))
            self.index.remove_ids(id_selector)

            self.save()
        else:
            print(f"The value '{value}' was not found in the index.")

    def get_embedding_by_index(self, index: int) -> np.ndarray:
        """
        Get the embedding corresponding to a certain index value.
        """
        if index < 0 or index >= len(self.values):
            raise ValueError("The index is out of range")

        embedding = self.index.reconstruct(index)

        return embedding

    def get_index_by_value(self, value: str) -> Optional[int]:
        """
        Get the index corresponding to a value in self.values.
        """
        if value in self.values:
            index = self.values.index(value)
            return index
        else:
            return None

    def get_embedding_by_value(self, value: str) -> Optional[np.ndarray]:
        """
        Get the embedding corresponding to a certain value in self.values.
        """
        index = self.get_index_by_value(value)
        if index is not None:
            embedding = self.get_embedding_by_index(index)
            return embedding
        else:
            return None

    def get_all_embeddings(self) -> np.ndarray:
        """
        Get all the embeddings in the index.
        """
        embeddings = []
        for i in range(len(self.values)):
            embeddings.append(self.get_embedding_by_index(i))
        self.embeddings = np.array(embeddings)
        return self.embeddings

    def faiss_query(self, query: str, k: int = 10) -> Tuple[List[str], List[float]]:
        """Query the faiss index for the top-k most similar values to the query"""

        embedding = self.embedder.embed(query)
        if k > len(self.values):
            k = len(self.values)
        D, I = self.index.search(np.array([embedding]).astype(np.float32), k)
        values = [self.values[i] for i in I[0]]
        scores = [d for d in D[0]]
        return values, scores, I

    def token_bound_query(self, query, k=10, max_tokens=4000):
        """Query the faiss index for the top-k most similar values to the query, but bound the number of tokens retrieved by the max_tokens parameter"""
        returned_tokens = 0
        top_k_hint = []
        scores = []
        tokens = []
        indices = []

        if len(self.values) > 0:
            top_k, scores, indices = self.faiss_query(query, k=min(k, len(self.values)))

            for hint in top_k:
                message_tokens = len(self.tokenizer.encode(hint))
                tokens.append(message_tokens)
                if returned_tokens + message_tokens <= max_tokens:
                    top_k_hint += [hint]
                    returned_tokens += message_tokens

            self.query_history.append(
                {
                    "query": query,
                    "hints": top_k_hint,
                    "scores": scores,
                    "indices": indices,
                    "hints_tokens": tokens,
                    "returned_tokens": returned_tokens,
                    "max_tokens": max_tokens,
                    "k": k,
                }
            )

        return top_k_hint, scores, indices

    def save(self):
        save(self)

    def load(self):
        load(self)

    def prune(
        self,
        constraint: Optional[str] = None,
        regex_pattern: Optional[str] = None,
        length_constraint: Optional[int] = None,
        tokenizer: Optional[tiktoken.Encoding] = None,
    ) -> "MemoryIndex":
        if tokenizer is None:
            tokenizer = self.tokenizer
        return prune_index(self, constraint, regex_pattern, length_constraint,tokenizer)