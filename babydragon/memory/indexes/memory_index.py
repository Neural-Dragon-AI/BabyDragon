import copy
import json
import os
import pickle
import re
import time
from typing import Dict, List, Optional, Tuple, Union

import faiss
import numpy as np
import tiktoken
from IPython.display import Markdown, display

from babydragon.models.embedders.ada2 import OpenAiEmbedder
import pandas as pd

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
    ):

        self.name = name
        self.embedder = OpenAiEmbedder()
        if save_path is None:
            save_path = "storage"

        self.save_path = save_path

        # Create the 'storage' folder if it does not exist
        os.makedirs(self.save_path, exist_ok=True)
        self.values = []
        if load is True:
            self.load()
        else:
            self.init_index(index, values, embeddings)
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
    ) -> None:

        """
        initializes the index, there are 4 cases:
        1. we create a new index from scratch
        2. we create a new index from a list of embeddings and values
        3. we create a new index from a faiss index and values list
        4. we load an index from a file
        """
        # fist case is when we create a new index from scratch
        if index is None and values is None and embeddings is None:
            print("Creating a new index")
            self.index = faiss.IndexFlatIP(self.embedder.get_embedding_size())
            self.values = []
        # second case is where we create the index from a list of embeddings
        elif (
            index is None
            and values is not None
            and embeddings is not None
            and len(values) == len(embeddings)
        ):
            print("Creating a new index from a list of embeddings and values")
            self.index = faiss.IndexFlatIP(self.embedder.get_embedding_size())
            for embedding, value in zip(embeddings, values):
                self.add_to_index(value, embedding)
        # third case is where we create the index from a faiss index and values list
        elif (
            isinstance(index, faiss.Index)
            and index.d == self.embedder.get_embedding_size()
            and type(values) == list
            and len(values) == index.ntotal
        ):
            print("Creating a new index from a faiss index and values list")
            self.index = index
            self.values = values
        # fourth case is where we create an index from a list of values, the values are embedded and the index is created
        elif index is None and values is not None and embeddings is None:
            print("Creating a new index from a list of values")
            self.index = faiss.IndexFlatIP(self.embedder.get_embedding_size())
            i = 0
            for value in values:
                # print the value id to see the progress
                print("Embedding value ", i, " of ", len(values))
                # start tracking the time using time
                start = time.time()
                self.add_to_index(value)
                # print the time it took to embed the value
                print("Embedding value ", i, " took ", time.time() - start, " seconds")
                i += 1
        else:
            raise ValueError(
                "The index is not a valid faiss index or the embedding dimension is not correct"
            )
        
    @classmethod
    def from_pandas(
        cls,
        data_frame: Union[pd.DataFrame, str],
        columns: Optional[Union[str, List[str]]] = None,
        name: str = 'memory_index',
        save_path: Optional[str] = None,
        in_place: bool = True,
        embeddings_col: Optional[str] = None,
    ) -> 'MemoryIndex':
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

        if isinstance(data_frame, str) and data_frame.endswith(".csv") and os.path.isfile(data_frame):
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
            raise ValueError("The data_frame is not a valid pandas dataframe or the columns are not valid or the path is not valid")

        values, embeddings = cls.extract_values_and_embeddings(data_frame, columns, embeddings_col)
        return cls(values=values, embeddings=embeddings, name=name, save_path=save_path)

    @staticmethod
    def extract_values_and_embeddings(
        data_frame: pd.DataFrame,
        columns: Union[str, List[str]],
        embeddings_col: Optional[str],
    ) -> Tuple[List[str], Optional[List[np.ndarray]]]:
        """
        Extract values and embeddings from a pandas DataFrame.

        Args:
            data_frame: The DataFrame to extract values and embeddings from.
            columns: The columns of the DataFrame to use as values.
            embeddings_col: The column name containing the embeddings.

        Returns:
            A tuple containing two lists: one with the extracted values and one with the extracted embeddings (if any).
        """

        if isinstance(columns, list) and len(columns) > 1:
            data_frame["values_combined"] = data_frame[columns].apply(lambda x: ' '.join(x), axis=1)
            columns = "values_combined"
        elif isinstance(columns, list) and len(columns) == 1:
            columns = columns[0]
        elif not isinstance(columns, str):
            raise ValueError("The columns are not valid")

        values = []
        embeddings = []

        for _, row in data_frame.iterrows():
            value = row[columns]
            values.append(value)

            if embeddings_col is not None:
                embedding = row[embeddings_col]
                embeddings.append(embedding)

        return values, embeddings if embeddings_col is not None else None

    def add_to_index(
        self,
        value: str,
        embedding: Optional[Union[List[float], np.ndarray, str]] = None,
        verbose: bool = True,
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
            if embedding is not None:
                if type(embedding) is list:
                    embedding = np.array([embedding])
                elif type(embedding) is str:
                    embedding = eval(embedding)
                    embedding = np.array([embedding]).astype(np.float32)
                elif type(embedding) is not np.ndarray:
                    raise ValueError("The embedding is not a valid type")

                # Ensure that the embedding is a 2D numpy array
                if embedding.ndim == 1:
                    embedding = embedding.reshape(1, -1)
                # print("embedding is ", embedding)
                # print("embedding type is ", type(embedding))
                # print("embedding shape is ", embedding.shape)
                self.index.add(embedding)
                self.values.append(value)
                self.save()
        else:
            if verbose:
                display(
                    Markdown(
                        "The value {value} was already in the index".format(value=value)
                    )
                )

    def get_embedding_by_index(self, index: int) -> np.ndarray:
        """
        Get the embedding corresponding to a certain index value.
        """
        if index < 0 or index >= len(self.values):
            raise ValueError("The index is out of range")

        # Fetch the embedding from the Faiss index
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

        # Embed the data
        embedding = self.embedder.embed(query)
        if k > len(self.values):
            k = len(self.values)
        # Query the Faiss index for the top-K most similar values
        D, I = self.index.search(np.array([embedding]).astype(np.float32), k)
        # Get the values corresponding to the indices
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
                # mark the message and gets the length in tokens
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
        """Save the index to disk using faiss and json and numpy"""
        # Create the directory to save the index, values, and embeddings
        save_directory = os.path.join(self.save_path, self.name)
        os.makedirs(save_directory, exist_ok=True)

        # Save the FAISS index
        index_filename = os.path.join(save_directory, f"{self.name}_index.faiss")
        faiss.write_index(self.index, index_filename)

        # Save the index values
        values_filename = os.path.join(save_directory, f"{self.name}_values.json")
        with open(values_filename, "w") as f:
            json.dump(self.values, f)

        # Save the numpy array of the embeddings
        embeddings_filename = os.path.join(
            save_directory, f"{self.name}_embeddings.npz"
        )
        # print(f"embs: {self.get_all_embeddings().shape}")
        np.savez_compressed(embeddings_filename, self.get_all_embeddings())

    def load(self):
        """Load the index, values, and embeddings from disk"""
        # Set the directory to load the index, values, and embeddings from
        load_directory = os.path.join(self.save_path, self.name)

        # Load the FAISS index
        index_filename = os.path.join(load_directory, f"{self.name}_index.faiss")
        self.index = faiss.read_index(index_filename)

        # Load the index values
        values_filename = os.path.join(load_directory, f"{self.name}_values.json")
        with open(values_filename, "r") as f:
            self.values = json.load(f)

        # Load the numpy array of the embeddings
        embeddings_filename = os.path.join(
            load_directory, f"{self.name}_embeddings.npz"
        )
        embeddings_data = np.load(embeddings_filename)
        self.embeddings = embeddings_data["arr_0"]

    def prune_index(
        self,
        constraint: Optional[str] = None,
        regex_pattern: Optional[str] = None,
        length_constraint: Optional[int] = None,
    ) -> "MemoryIndex":
        """Prune the index based on the constraint provided. Currently, only regex and length constraints are supported."""

        if constraint is not None:
            if constraint == "regex":
                if regex_pattern is None:
                    raise ValueError(
                        "regex_pattern must be provided for regex constraint."
                    )
                pruned_values, pruned_embeddings = self._prune_by_regex(regex_pattern)
            elif constraint == "length":
                if length_constraint is None:
                    raise ValueError(
                        "length_constraint must be provided for length constraint."
                    )
                pruned_values, pruned_embeddings = self._prune_by_length(
                    length_constraint
                )
            else:
                raise ValueError("Invalid constraint type provided.")
        else:
            raise ValueError("constraint must be provided for pruning the index.")

        # Create a new index with pruned values and embeddings
        pruned_memory_index = MemoryIndex(
            values=pruned_values,
            embeddings=pruned_embeddings,
            name=self.name + "_pruned",
        )

        return pruned_memory_index

    def _prune_by_regex(self, regex_pattern: str) -> Tuple[List[str], List[np.ndarray]]:
        """Prune the index by the regex pattern provided."""
        pruned_values = []
        pruned_embeddings = []

        for value in self.values:
            if re.search(regex_pattern, value):
                pruned_values.append(value)
                pruned_embeddings.append(self.get_embedding_by_value(value))

        return pruned_values, pruned_embeddings

    def _prune_by_length(
        self, length_constraint: int
    ) -> Tuple[List[str], List[np.ndarray]]:
        """Prune the index by the length constraint provided."""
        pruned_values = []
        pruned_embeddings = []

        for value in self.values:
            if len(value) >= length_constraint:
                pruned_values.append(value)
                pruned_embeddings.append(self.get_embedding_by_value(value))

        return pruned_values, pruned_embeddings
