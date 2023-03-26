import pandas as pd
import copy
import os
from babydragon.core.vector_index import MemoryIndex

class PandasIndex(MemoryIndex):
    def __init__(self, pandaframe, columns=None, name='panda_index', save_path=None, in_place=True, embeddings_col=None):
        self.columns = columns
        self.values = []

        # Load or copy pandaframe, and set self.name, self.columns
        if type(pandaframe) == str and pandaframe.endswith(".csv") and os.path.isfile(pandaframe):
            try:
                pandaframe = pd.read_csv(pandaframe)
            except:
                raise ValueError("The CSV file is not valid")
            self.name = pandaframe.split("/")[-1].split(".")[0]
            self.columns = "values"
        elif type(pandaframe) == pd.core.frame.DataFrame and columns is not None:
            if not in_place:
                pandaframe = copy.deepcopy(pandaframe)
        else:
            raise ValueError("The pandaframe is not a valid pandas dataframe or the columns are not valid or the path is not valid")

        values, embeddings = self.extract_values_and_embeddings(pandaframe, embeddings_col)
        super().__init__(values=values, embeddings=embeddings, name=name, save_path=save_path)

    def extract_values_and_embeddings(self, pandaframe, embeddings_col):
        if type(self.columns) == list and len(self.columns) > 1:
            pandaframe["values"] = pandaframe[self.columns].apply(lambda x: ' '.join(x), axis=1)
            self.columns = "values"
        elif type(self.columns) == list and len(self.columns) == 1:
            self.columns = self.columns[0]
            pandaframe["values"] = pandaframe[self.columns]
            self.columns = "values"
        elif type(self.columns) != str:
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

    