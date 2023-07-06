from babydragon.types import infer_embeddable_type
from typing import  List, Optional, Union
from babydragon.models.embedders.ada2 import OpenAiEmbedder
from babydragon.models.embedders.cohere import CohereEmbedder
import polars as pl
import numpy as np
class MemoryFrame:
    def __init__(self, df: pl.DataFrame,
                context_columns: List,
                embeddable_columns: List,
                time_series_columns: List,
                name: str = "memory_frame",
                save_path: Optional[str] = None,
                load: bool = False,
                text_embedder: Optional[Union[OpenAiEmbedder,CohereEmbedder]] = OpenAiEmbedder,
                markdown: str = "text/markdown",
                token_overflow_strategy: str = "ignore"):
        self.df = df
        self.context_columns = context_columns
        self.time_series_columns = time_series_columns
        self.embeddable_columns = embeddable_columns
        self.meta_columns = ['ID', 'Name', 'Source', 'Author', 'Created At', 'Last Modified At']
        self.embedding_columns = []
        self.name = name
        self.save_path = save_path
        self.load = load
        self.text_embedder = text_embedder
        self.markdown = markdown
        self.token_overflow_strategy = token_overflow_strategy

    def __getattr__(self, name):
        # delegate to the self.df object
        return getattr(self.df, name)

    def get_overwritten_attr(self):
        df_methods = [method for method in dir(self.df) if callable(getattr(self.df, method))]
        memory_frame_methods = [method for method in dir(MemoryFrame) if callable(getattr(MemoryFrame, method))]
        common_methods = list(set(df_methods) & set(memory_frame_methods))
        return common_methods

    def embed_columns(self, embeddable_columns):
        for column_name in embeddable_columns:
            column = self.df[column_name]
            _, embedder = infer_embeddable_type(column)
            self._embed_column(column, embedder)

    def _embed_column(self, column, embedder):
        # Add the embeddings as a new column
        # Generate new values
        new_values = embedder.embed(self.df[column.name].to_list())
        # Add new column to DataFrame
        new_series = pl.Series(new_column_name, new_values)
        self.df = self.df.with_columns(new_series)
        new_column_name = f'embedding|{column.name}'
        self.embedding_columns.append(new_column_name)


    def search_column_with_sql_polar(self, sql_query, query, embeddable_column_name, top_k):
        df = self.df.filter(sql_query)
        embedding_column_name = 'embedding|' + embeddable_column_name

        query_as_series = pl.Series(query)
        dot_product_frame = df.with_columns(df[embedding_column_name].list.eval(pl.element().explode().dot(query_as_series),parallel=True).list.first().alias("dot_product"))
        # Sort by dot product and select top_k rows
        result = dot_product_frame.sort('dot_product', descending=True).slice(0, top_k)
        return result


    def search_column_polar(self, query, embeddable_column_name, top_k):
        embedding_column_name = 'embedding|' + embeddable_column_name

        query_as_series = pl.Series(query)
        dot_product_frame = self.df.with_columns(self.df[embedding_column_name].list.eval(pl.element().explode().dot(query_as_series),parallel=True).list.first().alias("dot_product"))
        # Sort by dot product and select top_k rows
        result = dot_product_frame.sort('dot_product', descending=True).slice(0, top_k)
        return result

    def search_column_numpy(self, query, embeddable_column_name, top_k):
        embedding_column_name = 'embedding|' + embeddable_column_name
        #convert query and column to numpy arrays
        column_np = self.df[embedding_column_name].to_numpy()
        #calculate dot product
        dot_product = np.dot(column_np, query)
        #add dot products as column to dataframe
        dot_product_frame = self.df.with_columns(dot_product)
        # Sort by dot product and select top_k rows
        result = dot_product_frame.sort('dot_product', descending=True).slice(0, top_k)
        return result

    def save_parquet(self):
        #save to arrow
        self.full_save_path = self.save_path + self.name + '.parquet'
        self.df.write_parquet(self.full_save_path)

    def load_parquet(self):
        self.full_save_path = self.save_path + self.name + '.parquet'
        self.df = pl.read_parquet(self.full_save_path)


    def search_time_series_column(self, query, embeddable_column_name, top_k):
        ## uses dtw to match any sub-sequence of the query to the time series in the column
        ## time series column have a date o time or delta time column associated with them 
        ## each row-value is a list of across rows variable length for both the time series and the date or time column
        pass

    def generate_column(self, row_generator, new_column_name):
        # Generate new values
        new_values = row_generator.generate(self.df)
        # Add new column to DataFrame
        new_df = pl.DataFrame({ new_column_name: new_values })

        # Concatenate horizontally
        self.df = self.df.hstack([new_df])
    
    def create_stratas(self):
        """
        Creates stratas for all columns in the DataFrame by calling _create_strata on each column.
        """
        pass

    def _create_strata(self, column_name: str):
        """
        Determine the correct strata creation function to call based on the column's data type,
        and then calls the corresponding function.

        Args:
            column_name (str): The name of the column.
        """
        pass

    def _create_strata_from_categorical(self, column_name: str):
        """
        Create strata for a categorical column.

        Args:
            column_name (str): The name of the column.
        """
        pass

    def _create_strata_from_real(self, column_name: str):
        """
        Create strata for a real valued column.

        Args:
            column_name (str): The name of the column.
        """
        pass

    def _create_strata_from_embeddings(self, column_name: str):
        """
        Create strata for a column with embeddings.

        Args:
            column_name (str): The name of the column.
        """
        pass

    def _create_strata_from_episodic_time_series(self, column_name: str):
        """
        Create strata for a column with episodic time series.

        Args:
            column_name (str): The name of the column.
        """
        pass

    def create_joint_strata(self, column_names: list):
        """
        Create strata based on the unique combinations of values across given columns.
        
        Args:
            column_names (list): The names of the columns.
        """
        pass

    def stratified_sampling(self, strata_columns: list, n_samples: int):
        """
        Perform stratified sampling on given stratum columns.
        
        Args:
            strata_columns (list): The names of the stratum columns.
            n_samples (int): The number of samples to draw.
        """
        pass

    def stratified_cross_validation(self, strata_columns: list, n_folds: int):
        """
        Perform stratified cross-validation on given stratum columns.
        
        Args:
            strata_columns (list): The names of the stratum columns.
            n_folds (int): The number of folds for the cross-validation.
        """
        pass
