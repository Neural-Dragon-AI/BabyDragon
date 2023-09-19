import os
from typing import List, Optional

import libcst as cst
import polars as pl
from pydantic import BaseModel

from babydragon.memory.frames.base_frame import BaseFrame
from babydragon.memory.frames.frame_models import CodeFramePydantic
from babydragon.processors.parsers.visitors.module_augmenters import CodeReplacerVisitor
from babydragon.models.generators.PolarsGenerator import PolarsGenerator
from babydragon.utils.frame_generators import load_generated_content
from babydragon.utils.main_logger import logger
from babydragon.utils.pythonparser import (
    extract_values_python,
    traverse_and_collect_rtd,
)
from babydragon.processors.parsers.visitors.node_type_counters import *
from babydragon.processors.parsers.visitors.operator_counters import *


class CodeFrame(BaseFrame):
    def __init__(
        self,
        df: pl.DataFrame,
        context_columns: Optional[List[str]] = None,
        embeddable_columns: Optional[List[str]] = None,
        embedding_columns: Optional[List[str]] = None,
        name: str = "code_frame",
        save_path: Optional[str] = "./storage",
        markdown: str = "text/markdown",
    ):
        BaseFrame.__init__(
            self,
            context_columns=...,
            embeddable_columns=...,
            embedding_columns=...,
            name=...,
            save_path=...,
            markdown=...,
        )
        self.df = df
        self.context_columns = context_columns
        self.embeddable_columns = embeddable_columns
        self.embedding_columns = embedding_columns
        self.name = name
        self.save_path = save_path
        self.markdown = markdown
        self.save_dir = f"{self.save_path}/{self.name}"
        self.frame_template = CodeFramePydantic(
            df_path=f"{self.save_dir}/{self.name}.parquet",
            context_columns=self.context_columns,
            embeddable_columns=self.embeddable_columns,
            embedding_columns=self.embedding_columns,
            name=self.name,
            save_path=self.save_path,
            save_dir=self.save_dir,
            load=True,
            markdown=self.markdown,
        )

    def __getattr__(self, name: str):
        if "df" in self.__dict__:
            return getattr(self.df.lazy(), name)
        raise AttributeError(
            f"{self.__class__.__name__} object has no attribute {name}"
        )

    def get_overwritten_attr(self):
        df_methods = [
            method for method in dir(self.df) if callable(getattr(self.df, method))
        ]
        memory_frame_methods = [
            method for method in dir(CodeFrame) if callable(getattr(CodeFrame, method))
        ]
        common_methods = list(set(df_methods) & set(memory_frame_methods))
        return common_methods

    def tokenize_column(self, column_name: str):
        new_values = self.tokenizer.encode_batch(self.df[column_name].to_list())
        new_series = pl.Series(f"tokens|{column_name}", new_values)
        len_values = [len(x) for x in new_values]
        new_series_len = pl.Series(f"tokens_len|{column_name}", len_values)
        self.df = self.df.with_columns(new_series)
        self.df = self.df.with_columns(new_series_len)
        return self

    def apply_validator_to_column(self, column_name: str, validator: type):
        # Ensure the validator is a subclass of BaseModel from Pydantic
        if not issubclass(validator, BaseModel):
            raise TypeError("validator must be a subclass of BaseModel from Pydantic")
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
                raise ValueError(
                    f"Failed to validate text in column '{column_name}'."
                ) from e

        return self

    def prepare_column_for_embeddings(self, column_name: str = "content"):
        df = self.df.select(column_name).with_columns(
            pl.lit("text-embedding-ada-002").alias("model")
        )
        input_df = df.with_columns(df[column_name].alias("input")).drop(column_name)
        return input_df

    def embed_column(
        self, column: str = "content", generator_log_name: str = "code_embedding"
    ):
        input_df = self.prepare_column_for_embeddings(column)
        embedder = PolarsGenerator(
            input_df=input_df, name=f"{generator_log_name}_text-embedding-ada-002"
        )
        embedder.execute()
        out_path = (
            f"./batch_generator/{generator_log_name}_text-embedding-ada-002.ndjson"
        )
        output = load_generated_content(out_path)
        self.df = self.df.with_columns(output)

    def convert_column_to_messages(
        self,
        column_name: str,
        model_name="gpt-3.5-turbo-16k",
        system_prompt="Youre a Helpful Summarizer!",
    ):
        df = self.df.select(column_name).with_columns(pl.lit(model_name).alias("model"))

        def create_content(value):
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{value}"},
            ]

        input_df = df.with_columns(
            df[column_name]
            .apply(create_content, return_dtype=pl.List)
            .alias("messages")
        ).drop(column_name)
        self.df = self.df.with_columns(input_df)

    def generate_column(
        self,
        column_name: str,
        generator_log_name: str = "code_summary",
        model_name: str = "gpt-3.5-turbo-16k",
        system_prompt: str = "Youre a Helpful Summarizer!",
    ):
        # TODO: Generate column with OpenAI functionAPI
        self.convert_column_to_messages(
            column_name=column_name, model_name=model_name, system_prompt=system_prompt
        )
        generator = PolarsGenerator(input_df=self.df, name=generator_log_name)
        generator.execute()
        out_path = f"./batch_generator/{generator_log_name}_output.ndjson"
        output = load_generated_content(out_path)
        self.df = self.df.with_columns(output)

    def search_column_with_sql_polar(
        self, sql_query, query, embeddable_column_name, top_k
    ):
        df = self.df.filter(sql_query)
        embedding_column_name = "embedding|" + embeddable_column_name

        query_as_series = pl.Series(query)
        dot_product_frame = df.with_columns(
            df[embedding_column_name]
            .list.eval(pl.element().explode().dot(query_as_series), parallel=True)
            .list.first()
            .alias("dot_product")
        )
        # Sort by dot product and select top_k rows
        result = dot_product_frame.sort("dot_product", descending=True).slice(0, top_k)
        return result

    def search_column_polar(self, query, embeddable_column_name, top_k):
        embedding_column_name = "embedding|" + embeddable_column_name

        query_as_series = pl.Series(query)
        dot_product_frame = self.df.with_columns(
            self.df[embedding_column_name]
            .list.eval(pl.element().explode().dot(query_as_series), parallel=True)
            .list.first()
            .alias("dot_product")
        )
        # Sort by dot product and select top_k rows
        result = dot_product_frame.sort("dot_product", descending=True).slice(0, top_k)
        return result

    def save(self):
        # create dir in storage if not exists
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.full_save_path = f"{self.save_path}/{self.name}/{self.name}.parquet"
        self.df.write_parquet(self.full_save_path)
        frame_template_json = self.frame_template.json()
        with open(f"{self.save_dir}/{self.name}.json", "w") as f:
            f.write(frame_template_json)

    @classmethod
    def load(cls, frame_path: str, name: str):
        df = pl.read_parquet(f"{frame_path}/{name}.parquet")
        with open(f"{frame_path}/{name}.json", "r") as f:
            frame_template = CodeFramePydantic.parse_raw(f.read())
        return cls(
            df=df,
            context_columns=frame_template.context_columns,
            embeddable_columns=frame_template.embeddable_columns,
            embedding_columns=frame_template.embedding_columns,
            name=frame_template.name,
            save_path=frame_template.save_path,
            markdown=frame_template.markdown,
        )

    def apply_visitor_to_column(
        self,
        column_name: str,
        visitor_class: type,
        new_column_prefix: Optional[str] = None,
    ):
        # Ensure the visitor_class is a subclass of PythonCodeVisitor
        if not issubclass(visitor_class, cst.CSTVisitor):
            raise TypeError("visitor_class must be a subclass of PythonCodeVisitor")

        # Iterate over the specified column
        new_values = []
        for code in self.df[column_name]:
            # Create a visitor and apply it to the code
            visitor = visitor_class(code)
            new_value = visitor.collect()
            new_values.append(new_value)
        # Generate new column
        new_column_name = f"{column_name}_{new_column_prefix}|{visitor_class.__name__}"
        new_series = pl.Series(new_column_name, new_values)
        self.df = self.df.with_columns(new_series)

        return self

    def count_node_types(self, column_name: str, new_column_prefix: str = "node_count"):
        for node_type_counter in NODETYPE_COUNTERS:
            self.apply_visitor_to_column(
                column_name, globals()[node_type_counter], new_column_prefix
            )
        return self

    def count_operators(
        self, column_name: str, new_column_prefix: str = "operator_count"
    ):
        for operator_counter in OPERATOR_COUNTERS:
            self.apply_visitor_to_column(
                column_name, globals()[operator_counter], new_column_prefix
            )
        return self

    def replace_code_in_files(
        self,
        filename_column: str,
        original_code_column: str,
        replacing_code_column: str,
    ):
        visitor = CodeReplacerVisitor(
            filename_column, original_code_column, replacing_code_column
        )
        for row in self.df.rows():
            filename = row[filename_column]
            original_code = row[original_code_column]
            replacing_code = row[replacing_code_column]

            if (
                filename
                and original_code
                and replacing_code
                and os.path.isfile(filename)
            ):
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
        markdown: str = "text/markdown",
    ) -> "CodeFrame":
        values, context = extract_values_python(
            directory_path, minify_code, remove_docstrings, resolution
        )
        logger.info(f"Found {len(values)} values in the directory {directory_path}")
        # convert retrieved data to polars dataframe
        df = pl.DataFrame({value_column: values})
        context_df = pl.DataFrame(context)
        # merge context columns with dataframe
        df = pl.concat([df, context_df], how="horizontal")
        if value_column not in embeddable_columns:
            embeddable_columns.append(value_column)
        kwargs = {
            "context_columns": context_columns,
            "embeddable_columns": embeddable_columns,
            "embedding_columns": embeddings_column,
            "name": name,
            "save_path": save_path,
            "markdown": markdown,
        }
        return cls(df, **kwargs)

    @classmethod
    def from_documentation(
        cls,
        directory_path,
        value_column: str,
        embedding_columns: List[str] = [],
        embeddable_columns: List[str] = [],
        name: str = "documentation_frame",
        save_path: Optional[str] = "./storage",
        markdown: str = "text/markdown",
    ) -> "CodeFrame":
        # Get the data from the directory
        data = traverse_and_collect_rtd(directory_path)
        logger.info(f"Found {len(data)} values in the directory {directory_path}")
        # conver list of tuples to polars dataframe (filename, content)
        df = pl.DataFrame(data, schema={"filename": str, value_column: str})

        # Add a column with the filename without the extension
        if value_column not in embeddable_columns:
            embeddable_columns.append(value_column)
        context_columns = ["filename"]
        kwargs = {
            "context_columns": context_columns,
            "embeddable_columns": embeddable_columns,
            "embedding_columns": embedding_columns,
            "name": name,
            "save_path": save_path,
            "markdown": markdown,
        }
        return cls(df, **kwargs)
