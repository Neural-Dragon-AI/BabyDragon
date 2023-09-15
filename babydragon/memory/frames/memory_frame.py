from babydragon.bd_types import infer_embeddable_type

from typing import  List, Optional, Union
from babydragon.memory.frames.base_frame import BaseFrame
from babydragon.utils.main_logger import logger
from babydragon.utils.dataframes import extract_values_and_embeddings_pd, extract_values_and_embeddings_hf, extract_values_and_embeddings_polars, get_context_from_hf, get_context_from_pandas, get_context_from_polars
from datasets import load_dataset
import polars as pl
import os
from pydantic import ConfigDict, BaseModel

class MemoryFramePydantic(BaseModel):
    df_path: str
    context_columns: List[str]
    embeddable_columns: List[str]
    embedding_columns: List[str]
    time_series_columns: List[str]
    name: str
    save_path: Optional[str]
    save_dir: str
    markdown: str
    model_config = ConfigDict(arbitrary_types_allowed=True)


class MemoryFrame(BaseFrame):
    def __init__(self, df: pl.DataFrame,
                context_columns: List = [],
                embeddable_columns: List = [],
                embedding_columns: List = [],
                time_series_columns: List = [],
                name: str = "memory_frame",
                save_path: Optional[str] = None,
                markdown: str = "text/markdown",):
        BaseFrame.__init__(self, context_columns=..., embeddable_columns=..., embedding_columns=..., name=..., save_path=..., markdown=...)
        self.df = df
        self.context_columns = context_columns
        self.time_series_columns = time_series_columns
        self.embeddable_columns = embeddable_columns
        self.meta_columns = ['ID', 'Name', 'Source', 'Author', 'Created At', 'Last Modified At']
        self.embedding_columns = embedding_columns
        self.name = name
        self.save_path = save_path
        self.markdown = markdown


    def __getattr__(self, name: str):
        if "df" in self.__dict__:
            return getattr(self.df.lazy(), name)
        raise AttributeError(f"{self.__class__.__name__} object has no attribute {name}")

    def get_overwritten_attr(self):
        df_methods = [method for method in dir(self.df) if callable(getattr(self.df, method))]
        memory_frame_methods = [method for method in dir(MemoryFrame) if callable(getattr(MemoryFrame, method))]
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

    def convert_column_to_messages(self, column_name, model_name = "gpt-3.5-turbo-16k", system_prompt = "Youre a Helpful Summarizer!"):
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
            frame_template = MemoryFramePydantic.parse_raw(f.read())
        return cls(df=df, context_columns=frame_template.context_columns, embeddable_columns=frame_template.embeddable_columns, embedding_columns=frame_template.embedding_columns, name=frame_template.name, save_path=frame_template.save_path, text_embedder=frame_template.text_embedder, markdown=frame_template.markdown)


    def generate_column(self, row_generator, new_column_name):
        # Generate new values
        new_values = row_generator.generate(self.df)
        # Add new column to DataFrame
        new_df = pl.DataFrame({ new_column_name: new_values })

        # Concatenate horizontally
        self.df = self.df.hstack([new_df])
    

    @classmethod
    def from_hf_dataset(
        cls,
        dataset_url: str,
        value_column: str,
        data_split: str = "train",
        embeddings_column: Optional[str] = None,
        embeddable_columns: List = [],
        context_columns: Optional[List[str]] = None,
        time_series_columns: List = [],
        name: str = "memory_frame",
        save_path: Optional[str] = None,
        markdown: str = "text/markdown",
        token_overflow_strategy: str = "ignore",
    ) -> "MemoryFrame":
        dataset = load_dataset(dataset_url)[data_split]
        values, embeddings = extract_values_and_embeddings_hf(dataset, value_column, embeddings_column)
        if context_columns is not None:
            context = get_context_from_hf(dataset, context_columns)
        else:
            context = None
        #convert retrieved data to polars dataframe
        if embeddings is not None:
            df = pl.DataFrame({value_column: values, embeddings_column: embeddings})
        else:
            df = pl.DataFrame({value_column: values})
        context_df = pl.DataFrame(context)
        #merge context columns with dataframe
        df = pl.concat([df, context_df], how='horizontal')
        if value_column not in embeddable_columns:
            embeddable_columns.append(value_column)
        return cls(df, context_columns, embeddable_columns, time_series_columns, name, save_path, markdown, token_overflow_strategy)