import copy
import os
from typing import Callable, List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

from babydragon.memory.indexes.memory_index import MemoryIndex


class PandaIndex(MemoryIndex):
    """
    A class to create an index of a pandas DataFrame, allowing querying on specified columns.
    Inherits from MemoryIndex class.
    """

    def __init__(self, df: pd.DataFrame, row_func: Optional[Callable[[pd.Series], str]] = None, columns: Optional[List[str]] = None):
        """
        Initialize a PandaIndex object.
        
        Args:
            df: A pandas DataFrame to index.
            row_func: An optional function to process rows before adding them to the index.
            columns: An optional list of column names to index. By default, it will index all string columns and columns containing lists with a single string.
        """
        if row_func is None:
            row_func = lambda row: str(row)

        self.df = df
        super().__init__() # Initialize the parent MemoryIndex class
        
        # Initialize the row-wise index
        for _, row in df.iterrows():
            self.add_to_index(row_func(row))
        
        self.columns: Dict[str, MemoryIndex] = {} 

        # Set up columns during initialization
        if columns is None:
            self.setup_columns()
        else:
            self.setup_columns(columns)

    def setup_columns(self, columns: Optional[List[str]] = None):
        """
        Set up columns for indexing.
        
        Args:
            columns: An optional list of column names to index. By default, it will index all string columns and columns containing lists with a single string.
        """
        if columns is None:
            # Use string columns or columns with lists containing a single string by default
            columns = [col for col in self.df.columns if self.df[col].apply(lambda x: isinstance(x, str) or (isinstance(x, list) and len(x) == 1 and isinstance(x[0], str))).all()]

        for col in columns:
            self.columns[col] = MemoryIndex.from_pandas(self.df, columns=col)

    def query_columns(self, query: str, columns: List[str]) -> List[Tuple[str, float]]:
        """
        Query the indexed columns of the DataFrame.
        
        Args:
            query: The search query as a string.
            columns: A list of column names to query.
        
        Returns:
            A list of tuples containing the matched value and its similarity score.
        """
        results = []
        for col in columns:
            if col in self.columns:
                results.extend(self.columns[col].faiss_query(query))
            else:
                raise KeyError(f"Column '{col}' not found in PandaDb columns dictionary.")
        return results
