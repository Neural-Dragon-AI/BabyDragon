from abc import ABC, abstractmethod
from typing import List, Optional, Union
import tiktoken
class BaseFrame(ABC):
    def __init__(self,
                context_columns: List = [],
                embeddable_columns: List = [],
                embedding_columns: List = [],
                name: str = "base_frame",
                save_path: Optional[str] = "/storage",
                markdown: str = "text/markdown",):
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.meta_columns = ['ID', 'Name', 'Source', 'Author', 'Created At', 'Last Modified At']
        self.context_columns = context_columns
        self.embeddable_columns = embeddable_columns
        self.embedding_columns = embedding_columns
        self.name = name
        self.save_path = save_path
        self.save_dir = f'{self.save_path}/{self.name}'
        self.markdown = markdown

    @abstractmethod
    def __getattr__(self, name: str):
        pass

    @abstractmethod
    def get_overwritten_attr(self):
        pass

    @abstractmethod
    def embed_columns(self, embeddable_columns: List):
        pass

    @abstractmethod
    def _embed_column(self, column, embedder):
        pass

    @abstractmethod
    def save(self):
        pass

    @classmethod
    @abstractmethod
    def load(cls, frame_path, name):
        pass

    @abstractmethod
    def generate_column(self, row_generator, new_column_name):
        pass

    
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