from abc import ABC, abstractmethod
from typing import List, Optional, Union
from babydragon.models.embedders.ada2 import OpenAiEmbedder
from babydragon.models.embedders.cohere import CohereEmbedder
import tiktoken
class BaseFrame(ABC):
    def __init__(self,
                context_columns: List = [],
                embeddable_columns: List = [],
                embedding_columns: List = [],
                name: str = "base_frame",
                save_path: Optional[str] = "/storage",
                text_embedder: Optional[Union[OpenAiEmbedder,CohereEmbedder]] = OpenAiEmbedder,
                markdown: str = "text/markdown",):
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.meta_columns = ['ID', 'Name', 'Source', 'Author', 'Created At', 'Last Modified At']
        self.context_columns = context_columns
        self.embeddable_columns = embeddable_columns
        self.embedding_columns = embedding_columns
        self.name = name
        self.save_path = save_path
        self.save_dir = f'{self.save_path}/{self.name}'
        self.text_embedder = text_embedder
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
    def search_column_with_sql_polar(self, sql_query, query, embeddable_column_name, top_k):
        pass

    @abstractmethod
    def search_column_polar(self, query, embeddable_column_name, top_k):
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
