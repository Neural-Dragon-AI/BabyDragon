import copy
import os
from typing import Callable, List, Optional, Tuple, Dict, Union

import numpy as np
import pandas as pd

from babydragon.memory.indexes.memory_index import MemoryIndex


class PandasIndex(MemoryIndex):
    """
    A class to create an index of a pandas DataFrame, allowing querying on specified columns.
    Inherits from MemoryIndex class.
    """

    def __init__(self, df: pd.DataFrame, row_func: Optional[Callable[[pd.Series], str]] = None, name= "pandas_index", columns: Optional[List[str]] = None):
        """
        Initialize a PandasIndex object.
        
        Args:
            df: A pandas DataFrame to index.
            row_func: An optional function to process rows before adding them to the index.
            columns: An optional list of column names to index. By default, it will index all string columns and columns containing lists with a single string.
        """
        if row_func is None:
            row_func = lambda row: str(row)
        self.row_func = row_func

        self.df = df
        MemoryIndex.__init__(self,name=name) # Initialize the parent MemoryIndex class
        
        # Initialize the row-wise index
        for _, row in df.iterrows():
            self.add_to_index(row_func(row))
        
        self.columns: Dict[str, MemoryIndex] = {} 

        # Set up columns during initialization
        if columns is None:
            self.setup_columns()
        else:
            self.setup_columns(columns)
        self.save()
        for col in self.columns:
            self.columns[col].save()

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
            self.columns[col] = MemoryIndex.from_pandas(self.df, columns=col, name=f"{self.name}_{col}")

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
    
    def add_row(self, row: pd.Series) -> None:
        """
        Add a row to the DataFrame and update the row and column indexes.

        Args:
            row: A pandas Series representing the row to add.
        """
        self.df = pd.concat([self.df, row.to_frame().T], ignore_index=True)
        self.add_to_index(self.row_func(row))

        for col in self.columns:
            if col in row:
                self.columns[col].add_to_index(row[col])


    def remove_row(self, index: int) -> None:
        """
        Remove a row from the DataFrame and update the row and column indexes.

        Args:
            index: The index of the row to remove.
        """
        if 0 <= index < len(self.df):
            self.remove_from_index(index)

            for col in self.columns:
                self.columns[col].remove_from_index(index)

            self.df.drop(index, inplace=True)
            self.df.reset_index(drop=True, inplace=True)
        else:
            raise IndexError(f"Index {index} is out of bounds for DataFrame with length {len(self.df)}")
    
    def rows_from_value(self, value: Union[str, int, float], column: Optional[str] = None) -> pd.DataFrame:
        """
        Return all rows of the DataFrame that have a particular value in the row index or a column index.

        Args:
            value: The value to search for in the DataFrame.
            column: The name of the column to search in. If None, search in the row index.

        Returns:
            A pandas DataFrame containing the rows with the specified value.
        """
        if column is None:
            return self.df.loc[self.df.index == value]
        else:
            if column in self.df.columns:
                return self.df.loc[self.df[column] == value]
            else:
                raise KeyError(f"Column '{column}' not found in the DataFrame.")


