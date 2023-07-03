from babydragon.types import infer_embeddable_type
import polars as pl

class MemoryFrame:
    def __init__(self, df: pl.DataFrame, context_columns: list, embeddable_columns: list):
        self.df = df
        self.context_columns = context_columns
        self.embeddable_columns = embeddable_columns
        self.meta_columns = ['ID', 'Name', 'Source', 'Author', 'Created At', 'Last Modified At']
        self.embedding_columns = []

    def __getattr__(self, name):
        # delegate to the self.df object
        return getattr(self.df, name)

    def get_overwritten_attr(self):
        df_methods = [method for method in dir(self.df) if callable(getattr(self.df, method))]
        memory_frame_methods = [method for method in dir(MemoryFrame) if callable(getattr(MemoryFrame, method))]
        common_methods = list(set(df_methods) & set(memory_frame_methods))
        return common_methods

    def embed_columns(self, embeddable_columns):
        for column in embeddable_columns:
            column_type, embedder = infer_embeddable_type(column)
            self._embed_column(column, embedder)

    def _embed_column(self, column, embedder):
        # the implementation of this function will depend on how exactly you want to embed the column
        # the result should be added to self.df and the column name should be added to self.embedding_columns
        pass

    def search_column(self, query, embeddable_column_name, top_k):
        embedding_column_name = 'embedding|' + embeddable_column_name
        
        # Add the query as a new column in the DataFrame
        n_rows = len(self.df)
        query_df = pl.DataFrame({ 'query': [query] * n_rows })

        extended_df = self.df.hstack([query_df])

        # Compute dot product of the target column with the query column
        dot_prod = extended_df.with_columns(
            extended_df[embedding_column_name].dot(extended_df['query']).alias('dot_prod')
        )

        # Sort by dot product and select top_k rows
        result = dot_prod.sort('dot_prod', reverse=True).slice(0, top_k)
        
        return result

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
