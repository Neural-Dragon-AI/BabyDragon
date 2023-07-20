from babydragon.types.bd_types import infer_embeddable_type
from typing import  List, Optional, Union
from babydragon.models.embedders.ada2 import OpenAiEmbedder
from babydragon.models.embedders.cohere import CohereEmbedder
from babydragon.utils.main_logger import logger
from babydragon.utils.pythonparser import extract_values_python
from babydragon.processors.github_processors import GithubProcessor
from babydragon.memory.frames.visitors.module_augmenters import CodeReplacerVisitor
from babydragon.memory.frames.base_frame import BaseFrame
from babydragon.memory.frames.visitors.node_type_counters import *
from babydragon.memory.frames.visitors.operator_counters import *
import hdbscan
import umap
import polars as pl
import os
from pydantic import ConfigDict, BaseModel
import libcst as cst



class CodeFramePydantic(BaseModel):
    df_path: str
    context_columns: List
    embeddable_columns: List
    embedding_columns: List
    name: str
    save_path: Optional[str]
    save_dir: str
    text_embedder: Optional[Union[OpenAiEmbedder,CohereEmbedder]] = OpenAiEmbedder
    markdown: str
    model_config = ConfigDict(arbitrary_types_allowed=True)

class CodeFrame(BaseFrame):
    """
    @daniel
    remove kwargs and be explicit about what is passed in
    """
    def __init__(self, df: pl.DataFrame, **kwargs):
        super().__init__(**kwargs)
        self.df = df
        self.frame_template = CodeFramePydantic(df_path=f'{self.save_dir}/{self.name}.parquet', context_columns=self.context_columns, embeddable_columns=self.embeddable_columns, embedding_columns=self.embedding_columns, name=self.name, save_path=self.save_path, save_dir=self.save_dir, load=True, text_embedder=self.text_embedder, markdown=self.markdown)

    def __getattr__(self, name: str):
        if "df" in self.__dict__:
            return getattr(self.df.lazy(), name)
        raise AttributeError(f"{self.__class__.__name__} object has no attribute {name}")

    def get_overwritten_attr(self):
        df_methods = [method for method in dir(self.df) if callable(getattr(self.df, method))]
        memory_frame_methods = [method for method in dir(CodeFrame) if callable(getattr(CodeFrame, method))]
        common_methods = list(set(df_methods) & set(memory_frame_methods))
        return common_methods

    def tokenize_column(self, column_name: str):
        new_values = self.tokenizer.encode_batch(self.df[column_name].to_list())
        new_series = pl.Series(f'tokens|{column_name}', new_values)
        len_values = [len(x) for x in new_values]
        new_series_len = pl.Series(f'tokens_len|{column_name}', len_values)
        self.df = self.df.with_columns(new_series)
        self.df = self.df.with_columns(new_series_len)
        return self

    def embed_columns(self, embeddable_columns: List):
        for column_name in embeddable_columns:
            column = self.df[column_name]
            _, embedder = infer_embeddable_type(column)
            self._embed_column(column, embedder)

    def _embed_column(self, column, embedder):
        # Add the embeddings as a new column
        # Generate new values
        new_values = embedder.embed(self.df[column.name].to_list())
        # Add new column to DataFrame
        new_column_name = f'embedding|{column.name}'
        new_series = pl.Series(new_column_name, new_values)
        self.df = self.df.with_columns(new_series)
        self.embedding_columns.append(new_column_name)

    def apply_validator_to_column(self, column_name: str, validator: type):
        # Ensure the validator is a subclass of BaseModel from Pydantic
        if not issubclass(validator, BaseModel):
            raise TypeError('validator must be a subclass of BaseModel from Pydantic')
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' does not exist.")
        if column_name not in self.embeddable_columns:
            raise ValueError(f"Column '{column_name}' is not set to embeddable.")
        # Iterate over the specified column
        for text in self.df[column_name]:
            # Create a validator instance and validate the text
            try:
                _ = validator(text=text).text
            except Exception as e:
                raise ValueError(f"Failed to validate text in column '{column_name}'.") from e

        return self

    def convert_column_to_messages(self, column_name, model_name = "gpt-3.5-turbo-16k-0613", system_prompt = "Youre a Helpful Summarizer!"):
        df = self.df.select(column_name).with_columns(pl.lit(model_name).alias("model"))

        def create_content(value):
            return ([{"role": "system", "content":system_prompt},
                        {"role": "user", "content": f"{value}"}])

        input_df = df.with_columns(df[column_name].apply(create_content, return_dtype=pl.List).alias('messages')).drop(column_name)
        self.df = self.df.with_columns(input_df)
        return self

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

    def save(self):
        #create dir in storage if not exists
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.full_save_path = f'{self.save_path}/{self.name}/{self.name}.parquet'
        self.df.write_parquet(self.full_save_path)
        frame_template_json = self.frame_template.json()
        with open(f'{self.save_dir}/{self.name}.json', 'w') as f:
            f.write(frame_template_json)

    @classmethod
    def load(cls, frame_path, name):
        df = pl.read_parquet(f'{frame_path}/{name}.parquet')
        with open(f'{frame_path}/{name}.json', 'r') as f:
            frame_template = CodeFramePydantic.parse_raw(f.read())
        return cls(df=df, context_columns=frame_template.context_columns, embeddable_columns=frame_template.embeddable_columns, embedding_columns=frame_template.embedding_columns, name=frame_template.name, save_path=frame_template.save_path, text_embedder=frame_template.text_embedder, markdown=frame_template.markdown)

    def generate_column(self, row_generator, new_column_name):
        # Generate new values
        new_values = row_generator.generate(self.df)
        # Add new column to DataFrame
        new_df = pl.DataFrame({ new_column_name: new_values })
        # Concatenate horizontally
        self.df = self.df.hstack([new_df])

    def apply_visitor_to_column(self, column_name: str, visitor_class: cst.CSTVisitor, new_column_prefix: Optional[str] = None):
        # Ensure the visitor_class is a subclass of PythonCodeVisitor
        if not issubclass(visitor_class, cst.CSTVisitor):
            raise TypeError('visitor_class must be a subclass of PythonCodeVisitor')

        # Iterate over the specified column
        new_values = []
        for code in self.df[column_name]:
            # Create a visitor and apply it to the code
            visitor = visitor_class(code)
            new_value = visitor.collect()
            new_values.append(new_value)
        # Generate new column
        new_column_name = f'{column_name}_{new_column_prefix}|{visitor_class.__name__}'
        new_series = pl.Series(new_column_name, new_values)
        self.df = self.df.with_columns(new_series)

        return self

    def apply_code_transformer(self, column_name: str, transformer_class: cst.CSTTransformer, new_column_prefix: Optional[str] = None):
        """@daniel implement this function"""
        pass

    
    def count_node_types(self, column_name: str, new_column_prefix: str = 'node_count'):
        for node_type_counter in NODETYPE_COUNTERS:
            self.apply_visitor_to_column(column_name, globals()[node_type_counter], new_column_prefix)
        return self

    def count_operators(self, column_name: str, new_column_prefix: str = 'operator_count'):
        for operator_counter in OPERATOR_COUNTERS:
            self.apply_visitor_to_column(column_name, globals()[operator_counter], new_column_prefix)
        return self

    def cluster_embeddings(self, column_name: str, dim_reduction_model, cluster_model):
        embeddings = self.df[column_name].to_list()
        if dim_reduction_model is None:
            dim_reduction_model = umap.UMAP()
        if cluster_model is None:
            cluster_model = hdbscan.HDBSCAN()
        reduced_embeddings = dim_reduction_model.fit_transform(embeddings)
        labels = cluster_model.fit_predict(reduced_embeddings)
        new_column_name = f'cluster|{column_name}'
        new_series = pl.Series(new_column_name, labels)
        self.df = self.df.with_columns(new_series)
        return self

    def replace_code_in_files(self, filename_column: str, original_code_column: str, replacing_code_column: str):
        """@daniel here a  libcst codemod which uses the CodeReplacerVisitor should be applied instead of this function"""
        visitor = CodeReplacerVisitor(filename_column, original_code_column, replacing_code_column)
        for row in self.df.rows():
            filename = row[filename_column]
            original_code = row[original_code_column]
            replacing_code = row[replacing_code_column]

            if filename and original_code and replacing_code and os.path.isfile(filename):
                node = cst.parse_module(original_code)
                node.metadata[original_code_column] = original_code
                node.metadata[replacing_code_column] = replacing_code
                node.metadata[filename_column] = filename

                modified_node = node.visit(visitor)
                modified_code = cst.Module(body=modified_node.body).code
                row[original_code_column] = modified_code

        return self

    @classmethod
    def from_python(
        cls,
        directory_path: str,
        value_column: str,
        minify_code: bool = False,
        remove_docstrings: bool = False,
        resolution: str = "both",
        embeddings_column: List = [],
        embeddable_columns: List = [],
        context_columns: Optional[List[str]] = None,
        name: str = "code_frame",
        save_path: Optional[str] = "./storage",
        embedder: Optional[Union[OpenAiEmbedder,CohereEmbedder]]= None,
        markdown: str = "text/markdown",
    ) -> "CodeFrame":
        values, context = extract_values_python(directory_path, minify_code, remove_docstrings, resolution)
        logger.info(f"Found {len(values)} values in the directory {directory_path}")
        #convert retrieved data to polars dataframe
        df = pl.DataFrame({value_column: values})
        context_df = pl.DataFrame(context)
        #merge context columns with dataframe
        df = pl.concat([df, context_df], how='horizontal')
        if value_column not in embeddable_columns:
            embeddable_columns.append(value_column)
        print(type(embedder))
        kwargs = {
            "context_columns": context_columns,
            "embeddable_columns": embeddable_columns,
            "embedding_columns": embeddings_column,
            "name": name,
            "save_path": save_path,
            "text_embedder": embedder,
            "markdown": markdown
        }
        return cls(df, **kwargs)

