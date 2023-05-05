import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Union

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
            data_frame["values_combined"] = data_frame[columns].apply(
                lambda x: " ".join(x), axis=1
            )
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