import copy
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from babydragon.memory.indexes.memory_index import MemoryIndex


class PandasIndex(MemoryIndex):
    def __init__(
        self,
        pandaframe: Union[pd.DataFrame, str],
        columns: Optional[Union[str, List[str]]] = None,
        name: str = "panda_index",
        save_path: Optional[str] = None,
        in_place: bool = True,
        embeddings_col: Optional[str] = None,
    ):
        """
        Create a PandasIndex object.

        Args:
            pandaframe: The DataFrame or path to a CSV file.
            columns: The columns of the DataFrame to use as values.
            name: The name of the index.
            save_path: The path to save the index.
            in_place: Whether to work on the DataFrame in place or create a copy.
            embeddings_col: The column name containing the embeddings.
        """
        self.columns = columns
        self.values = []

        # Load or copy pandaframe, and set self.name, self.columns
        if (
            isinstance(pandaframe, str)
            and pandaframe.endswith(".csv")
            and os.path.isfile(pandaframe)
        ):
            try:
                pandaframe = pd.read_csv(pandaframe)
            except:
                raise ValueError("The CSV file is not valid")
            self.name = pandaframe.split("/")[-1].split(".")[0]
            self.columns = "values"
        elif isinstance(pandaframe, pd.core.frame.DataFrame) and columns is not None:
            if not in_place:
                pandaframe = copy.deepcopy(pandaframe)
        else:
            raise ValueError(
                "The pandaframe is not a valid pandas dataframe or the columns are not valid or the path is not valid"
            )

        values, embeddings = self.extract_values_and_embeddings(
            pandaframe, embeddings_col
        )
        super().__init__(
            values=values, embeddings=embeddings, name=name, save_path=save_path
        )

    def extract_values_and_embeddings(
        self,
        pandaframe: pd.DataFrame,
        embeddings_col: Optional[str],
    ) -> Tuple[List[str], Optional[List[np.ndarray]]]:
        """
        Extract values and embeddings from a pandas DataFrame.

        Args:
            pandaframe: The DataFrame to extract values and embeddings from.
            embeddings_col: The column name containing the embeddings.

        Returns:
            A tuple containing two lists: one with the extracted values and one with the extracted embeddings (if any).
        """
        if isinstance(self.columns, list) and len(self.columns) > 1:
            pandaframe["values"] = pandaframe[self.columns].apply(
                lambda x: " ".join(x), axis=1
            )
            self.columns = "values"
        elif isinstance(self.columns, list) and len(self.columns) == 1:
            self.columns = self.columns[0]
            pandaframe["values"] = pandaframe[self.columns]
            self.columns = "values"
        elif not isinstance(self.columns, str):
            raise ValueError("The columns are not valid")

        values = []
        embeddings = []

        for _, row in pandaframe.iterrows():
            value = row["values"]
            values.append(value)

            if embeddings_col is not None:
                embedding = row[embeddings_col]
                embeddings.append(embedding)

        return values, embeddings if embeddings_col is not None else None
