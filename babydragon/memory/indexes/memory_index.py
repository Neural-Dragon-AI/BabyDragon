from typing import List, Optional, Union, Tuple, Dict, Any
from babydragon.models.embedders.ada2 import OpenAiEmbedder
from babydragon.models.embedders.cohere import CohereEmbedder
from babydragon.memory.indexes.numpy_index import NpIndex
import numpy as np
import pandas as pd
import polars as pl
import os
from babydragon.utils.main_logger import logger
from babydragon.utils.dataframes import extract_values_and_embeddings_pd, extract_values_and_embeddings_hf, extract_values_and_embeddings_polars, get_context_from_hf, get_context_from_pandas, get_context_from_polars
from babydragon.utils.pythonparser import extract_values_and_embeddings_python
from datasets import load_dataset



class MemoryIndex(NpIndex):
    """
    this class is a wrapper for a Np index, it contains information about the format of the index the index itself
    ways to load it from: python lists, pandas dataframe, huggingface dataset, polars dataframe or local python package with libcst pre-processing
    a concept of context that can be used to store information about the values, there is a one to many relationship between values and context, 
    when loading from a dataframe or dataset the context is automatically extracted from the dataframe/dataset if context_columns are provided
    """
    @staticmethod
    def check_uniform_context_type(context: List[Any]) -> None:
        """Check if all context elements are of the same type."""
        if not all(isinstance(x, type(context[0])) for x in context):
            raise ValueError("All context elements must be of the same type.")

    def __init__(
        self,
        values: Optional[List[str]] = None,
        embeddings: Optional[List[Union[List[float], np.ndarray]]] = None,
        context: Optional[List[Any]] = None,
        name: str = "memory_index",
        save_path: Optional[str] = None,
        load: bool = False,
        embedder: Optional[Union[OpenAiEmbedder,CohereEmbedder]] = OpenAiEmbedder,
        markdown: str = "text/markdown",
        token_overflow_strategy: str = "ignore",
    ):
        NpIndex.__init__(self, values=values, embeddings=embeddings, name=name, save_path=save_path, load=load, embedder=embedder, token_overflow_strategy=token_overflow_strategy)
        if context is not None and len(context) != len(values):
            raise ValueError("The context must have the same length as the values")


        self.markdown = markdown
        if context is not None and values is not None:
            self.context = {value: [context[old_id] for old_id in self.old_ids[value]] for value in self.values}

        if context is not None:
            self.check_uniform_context_type(context)
            self.context_type = type(context[0])


    def get_context(self, identifier: Union[int, str, np.ndarray, List[Union[int, str, np.ndarray]]]) -> Optional[Any]:
        """ get the context of a value id or embedding or a list of """
        if isinstance(identifier, list):
            return [self.get_context(value) for value in identifier]
        else:
            id = self.identify_input(identifier)
            value = self.values[id]
            return self.context[value]

    def clean_context(self):
        """ method to be called after parent modifications with add/remove/update remove from context all the values that are not in the index anymore """
        self.context = {value: self.context[value] for value in self.values}

    def add_to_context(self, value: str, context: Any):
        """ add a context to a value """
        if not isinstance(context, self.context_type):
            raise ValueError("The context must be of the same type as the other contexts")
        if value in self.values:
            if value not in self.context:
                self.context[value] = []
            self.context[value].append(context)

    def add(self, values: List[str], embedding: Optional[Union[List[float], np.ndarray]] = None, context: Optional[Any] = None):
        """ add a value to the index, if the value is already in the index it will be updated """
        if isinstance(values, str):
            values = [values]
        NpIndex.add(self, values, embedding)
        if context is not None:
            for value, cont in zip(values, context):
                self.add_to_context(value, cont)

    def remove(self, identifier: Union[int, str, np.ndarray, List[Union[int, str, np.ndarray]]]) -> None:
        if not isinstance(identifier, list):
            id = self.identify_input(identifier)
            value = self.values[id]
        NpIndex.remove(self, identifier)
        if not isinstance(identifier, list):
            self.context.pop(value)

    def update(self, old_identifier: Union[int, str, np.ndarray, List[Union[int, str, np.ndarray]]], new_value: Union[str, List[str]], new_context: Optional[Any] = None, new_embedding: Optional[Union[List[float], np.ndarray, List[List[float]], List[np.ndarray]]] = None) -> None:
        #recover value from old_identifier
        if new_context is not None:
            if not isinstance(new_context, self.context_type):
                raise ValueError("The context must be of the same type as the other contexts")
        if not isinstance(old_identifier, list):
            old_id = self.identify_input(old_identifier)
            old_value = self.values[old_id]
            # Only perform the update if the old_value is not the same as the new_value.
            if old_value != new_value:
                NpIndex.update(self, old_identifier, new_value, new_embedding)

            if new_context is not None:
                self.context[new_value] = [new_context]
            else:
                self.context[new_value] = self.context.pop(old_value)
        else:
            self.context[new_value] = self.context.pop(old_value)

    @classmethod
    def from_pandas(
        cls,
        data_frame: Union[pd.DataFrame, str],
        value_column: str,
        embeddings_column: Optional[str] = None,
        context_columns: Optional[List[str]] = None,
        name: str = "memory_index",
        save_path: Optional[str] = None,
        embedder: Optional[Union[OpenAiEmbedder,CohereEmbedder]]= OpenAiEmbedder,
        markdown: str = "text/markdown",
    ) -> "MemoryIndex":
        if (
            isinstance(data_frame, str)
            and data_frame.endswith(".csv")
            and os.path.isfile(data_frame)
        ):
            logger.info("Loading the CSV file")
            data_frame = pd.read_csv(data_frame)
            name = os.path.basename(data_frame).split(".")[0]
        elif isinstance(data_frame, pd.core.frame.DataFrame):
            logger.info("Loading the pandas DataFrame")
        else:
            raise ValueError("The data_frame is not a valid pandas dataframe or the path is not valid")

        values, embeddings = extract_values_and_embeddings_pd(data_frame, value_column, embeddings_column)
        if context_columns is not None:
            context = get_context_from_pandas(data_frame, context_columns)

        return cls(values=values, embeddings=embeddings, name=name, save_path=save_path,markdown=markdown, embedder=embedder, context = context)

    @classmethod
    def from_hf_dataset(
        cls,
        dataset_url: str,
        value_column: str,
        data_split: str = "train",
        embeddings_column: Optional[str] = None,
        context_columns: Optional[List[str]] = None,
        name: str = "memory_index",
        save_path: Optional[str] = None,
        embedder: Optional[Union[OpenAiEmbedder,CohereEmbedder]]= OpenAiEmbedder,
        markdown: str = "text/markdown",
    ) -> "MemoryIndex":
        dataset = load_dataset(dataset_url)[data_split]
        values, embeddings = extract_values_and_embeddings_hf(dataset, value_column, embeddings_column)
        if context_columns is not None:
            context = get_context_from_hf(dataset, context_columns)
        return cls(values=values, embeddings=embeddings, name=name, save_path=save_path, embedder=embedder, markdown=markdown, context = context)

    @classmethod
    def from_polars(
        cls,
        data_frame: pl.DataFrame,
        value_column: str,
        embeddings_column: Optional[str] = None,
        context_columns: Optional[List[str]] = None,
        name: str = "memory_index",
        save_path: Optional[str] = None,
        embedder: Optional[Union[OpenAiEmbedder,CohereEmbedder]]= OpenAiEmbedder,
        markdown: str = "text/markdown",
    ) -> "MemoryIndex":
        print("Loading the Polars DataFrame")
        values, embeddings = extract_values_and_embeddings_polars(data_frame, value_column, embeddings_column)
        if context_columns is not None:
            context = get_context_from_polars(data_frame, context_columns)

        return cls(values=values, embeddings=embeddings, name=name, save_path=save_path,markdown=markdown, embedder=embedder, context = context)


    @classmethod
    def from_python(
        cls,
        directory_path: str,
        minify_code: bool = False,
        remove_docstrings: bool = False,
        name: str = "memory_index",
        save_path: Optional[str] = None,
        markdown: str = "python/markdown",
        resolution: str = "both",
        embedder: Optional[Union[OpenAiEmbedder,CohereEmbedder]]= OpenAiEmbedder,
    ) -> "MemoryIndex":
        values, context = extract_values_and_embeddings_python(directory_path, minify_code, remove_docstrings, resolution)
        logger.info(f"Found {len(values)} values in the directory {directory_path}")
        return cls(values=values, embeddings=None, name=name, save_path=save_path,markdown=markdown, embedder=embedder, context = context)