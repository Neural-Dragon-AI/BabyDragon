
import numpy as np
from typing import List, Optional, Tuple, Union, Dict, Any

import datasets

def concat_columns(example, index=None):
    column1='title'
    column2='text'
    example['merged_column'] = example[column1] +" - " + example[column2]
    return example

def extract_values_and_embeddings_hf(
    dataset: datasets.Dataset,
    value_column: Union[str, List[str]],
    embeddings_column: Optional[str],
) -> Tuple[List[str], Optional[List[np.ndarray]]]:
    """
    Extract values and embeddings from a Hugging Face dataset.

    Args:
        dataset: The Hugging Face dataset to extract values and embeddings from.
        value_column: The column(s) of the dataset to use as values.
        embeddings_column: The column name containing the embeddings.

    Returns:
        A tuple containing two lists: one with the extracted values and one with the extracted embeddings (if any).
    """
    if isinstance(value_column, str):
        value_column = [value_column]
    print("Merging values: Start")
    merged_docs = dataset.map(concat_columns, with_indices=True)
    print("Merging values: Done")
    values = merged_docs['merged_column']
    embeddings = dataset[embeddings_column]
    return values, embeddings if embeddings_column is not None else None

def extract_values_hf(dataset: datasets.Dataset, value_column: Union[str, List[str]]) -> List[str]:
    """
    Extract values from a Hugging Face dataset.

    Args:
        dataset: The Hugging Face dataset to extract values from.
        value_column: The column(s) of the dataset to use as values.

    Returns:
        A list with the extracted values.
    """
    if isinstance(value_column, str):
        value_column = [value_column]
    if len(value_column) == 1:
        return dataset[value_column[0]]
    elif len(value_column) > 1:
        merged_docs = dataset.map(concat_columns)
        return merged_docs
    else:
        raise ValueError("No value column specified.")

