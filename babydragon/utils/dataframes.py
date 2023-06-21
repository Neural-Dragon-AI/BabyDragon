import pandas as pd
import datasets
import polars as pl
import numpy as np
from typing import List, Optional, Tuple, Union, Dict, Any

def extract_values_and_embeddings_pd(
        data_frame: pd.DataFrame,
        value_column: str,
        embeddings_col: Optional[str],
    ) -> Tuple[List[str], Optional[List[np.ndarray]]]:
        """
        Extract values and embeddings from a pandas DataFrame.

        Args:
            data_frame: The DataFrame to extract values and embeddings from.
            value_column: The column of the DataFrame to use as values.
            embeddings_col: The column name containing the embeddings.

        Returns:
            A tuple containing two lists: one with the extracted values and one with the extracted embeddings (if any).
        """
        values = data_frame[value_column].tolist()
        embeddings = data_frame[embeddings_col].tolist() if embeddings_col else None

        return values, embeddings


def extract_values_and_embeddings_hf(
    dataset: datasets.Dataset,
    value_column: str,
    embeddings_column: Optional[str],
) -> Tuple[List[str], Optional[List[np.ndarray]]]:
    """
    Extract values and embeddings from a Hugging Face dataset.

    Args:
        dataset: The Hugging Face dataset to extract values and embeddings from.
        value_column: The column of the dataset to use as values.
        embeddings_column: The column name containing the embeddings.

    Returns:
        A tuple containing two lists: one with the extracted values and one with the extracted embeddings (if any).
    """
    values = dataset[value_column]
    embeddings = dataset[embeddings_column] if embeddings_column else None

    return values, embeddings

def extract_values_and_embeddings_polars(
        data_frame: pl.DataFrame,
        value_column: str,
        embeddings_column: Optional[str]
    ) -> Tuple[List[str], Optional[List[np.ndarray]]]:
    """
    Extract values and embeddings from a Polars DataFrame.

    Args:
        data_frame: The DataFrame to extract values and embeddings from.
        value_column: The column of the DataFrame to use as values.
        embeddings_column: The column name containing the embeddings.

    Returns:
        A tuple containing two lists: one with the extracted values and one with the extracted embeddings (if any).
    """
    values = data_frame[value_column].to_list()
    embeddings = data_frame[embeddings_column].to_list() if embeddings_column else None

    return values, embeddings

def get_context_from_pandas(
          data_frame: pd.DataFrame,
          context_columns: List[str]):
    """Extract context information from a pandas DataFrame."""
    # This function will convert the row of the DataFrame into a dictionary where the keys are the column names.
    def row_to_dict(row: pd.Series) -> Dict[str, Any]:
        return {column: row[column] for column in context_columns}

    # Apply the function to each row of the DataFrame and convert the result into a list.
    context = data_frame.apply(row_to_dict, axis=1).tolist()

    return context

def get_context_from_hf(
          data_frame: datasets.Dataset,
          context_columns: List[str]):
    """return a list dictionaries with the keys the column name and value the context columns values"""
    context_columns = {}
    context = []
    for column in context_columns:
        context_columns[column] = data_frame[column]
    data_frame_len = len(data_frame)
    for row in range(data_frame_len):
         context.append({column: context_columns[column][row] for column in context_columns})
    return context

def get_context_from_polars(
          data_frame: pl.DataFrame,
          context_columns: List[str]):
    """ return a list dictionaries with the keys the column name and value the context columns values"""
    pass

