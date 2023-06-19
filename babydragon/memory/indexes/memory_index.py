from typing import List, Optional, Union, Tuple, Dict, Any
from babydragon.models.embedders.ada2 import OpenAiEmbedder
from babydragon.models.embedders.cohere import CohereEmbedder
from babydragon.memory.indexes.numpy_index import NpIndex
import numpy as np
import pandas as pd
import polars as pl
import os
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
    ):
        if context is not None and len(context) != len(values):
            raise ValueError("The context must have the same length as the values")
        
        NpIndex.__init__(self, values, embeddings, name, save_path, load, embedder)
        self.markdown = markdown
        if context is not None and values is not None:
            self.context = {value: [context[old_id] for old_id in self.old_ids[value]] for value in self.values} 

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
        if value in self.values and value not in self.context:
            self.context[value] = []
        elif value in self.values:
            self.context[value].append(context)

    def add(self, values: List[str], embedding: Optional[Union[List[float], np.ndarray]] = None, context: Optional[Any] = None):
        """ add a value to the index, if the value is already in the index it will be updated """
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

    def update(self, old_identifier: Union[int, str, np.ndarray, List[Union[int, str, np.ndarray]]], new_value: Union[str, List[str]], new_embedding: Optional[Union[List[float], np.ndarray, List[List[float]], List[np.ndarray]]] = None) -> None:
        #recover value from old_identifier
        NpIndex.update(self, old_identifier, new_value, new_embedding)
        if not isinstance(old_identifier, list):
            old_id = self.identify_input(old_identifier)
            self.context[new_value] = self.context.pop(self.values[old_id])

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
            print("Loading the CSV file")
            data_frame = pd.read_csv(data_frame)
            name = os.path.basename(data_frame).split(".")[0]
        elif isinstance(data_frame, pd.core.frame.DataFrame):
            print("Loading the DataFrame")
        else:
            raise ValueError("The data_frame is not a valid pandas dataframe or the path is not valid")

        values, embeddings = extract_values_and_embeddings_pd(data_frame, value_column, embeddings_column)
        #for daniel here the get_context methods do not exists just inspiration for you do the same for hf datasets, polars and python
        if context_columns is not None:
            context = get_context_from_pandas(data_frame, context_columns)

        return cls(values=values, embeddings=embeddings, name=name, save_path=save_path,markdown=markdown, embedder=embedder, context = context)

    @classmethod
    def from_hf_dataset(
        cls,
        dataset_url: str,
        value_column: str,
        embeddings_column: Optional[str] = None,
        context_columns: Optional[List[str]] = None,
        name: str = "memory_index",
        save_path: Optional[str] = None,
        embedder: Optional[Union[OpenAiEmbedder,CohereEmbedder]]= OpenAiEmbedder,
        markdown: str = "text/markdown"
    ) -> "MemoryIndex":
        dataset = load_dataset(dataset_url)['train']
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
        resolution: str = "function",
        embedder: Optional[Union[OpenAiEmbedder,CohereEmbedder]]= OpenAiEmbedder,
    ) -> "MemoryIndex":
        values,embeddings, context = extract_values_and_embeddings_python(directory_path, minify_code, remove_docstrings, resolution)
        return cls(values=values, embeddings=embeddings, name=name, save_path=save_path,markdown=markdown, embedder=embedder, context = context)