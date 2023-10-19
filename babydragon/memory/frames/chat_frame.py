import os
from typing import Any, List

import polars as pl

from babydragon.memory.threads.base_thread import BaseThread
from babydragon.models.generators.PolarsGenerator import PolarsGenerator
from babydragon.utils.frame_generators import (
    SummaryColumnProcessor,
    TopicLabelProcessor,
    cluster_summaries,
    embed_input_column,
    load_generated_content,
)


class ChatFrame(BaseThread):
    def __init__(
        self,
        name: str = "chat_frame",
        context_columns: List[str] = [],
        embeddable_columns: List[str] = ["content"],
        embedding_columns: List[str] = [],
        markdown: str = "text/markdown",
        max_memory: int | None = None,
        tokenizer: Any | None = None,
        save_path: str = "threads",
    ) -> None:
        BaseThread.__init__(self, name, max_memory, tokenizer, save_path)
        self.context_columns = context_columns
        self.embeddable_columns = embeddable_columns
        self.embedding_columns = embedding_columns
        self.markdown = markdown

    def apply_expression(self, expression: pl.Expr) -> pl.DataFrame:
        return self.memory_thread.filter(expression)

    # Dot Product Query
    def search_column_with_dot_product(
        self, query: str, embeddable_column_name: str, top_k: int
    ) -> pl.DataFrame:
        embedding_column_name = "embedding|" + embeddable_column_name
        query_as_series = pl.Series(query)
        dot_product_frame = self.memory_thread.with_columns(
            self.memory_thread[embedding_column_name]
            .list.eval(pl.element().explode().dot(query_as_series), parallel=True)
            .list.first()
            .alias("dot_product")
        )
        # Sort by dot product and select top_k rows
        result = dot_product_frame.sort("dot_product", descending=True).slice(0, top_k)
        return result

    # Tokenization
    def tokenize_column(
        self, input_df: pl.DataFrame = None, column_name: str = "content"
    ):
        if input_df is None:
            input_df = self.memory_thread
        new_values = self.tokenizer.encode_batch(input_df[column_name].to_list())
        new_series = pl.Series(f"tokens|{column_name}", new_values)
        len_values = [len(x) for x in new_values]
        new_series_len = pl.Series(f"tokens_len|{column_name}", len_values)
        input_df = input_df.with_columns(new_series)
        input_df = input_df.with_columns(new_series_len)
        if input_df is None:
            self.memory_thread = input_df
            return self.memory_thread
        else:
            return input_df

    def prepare_column_for_embeddings(self, column_name: str = "content"):
        df = self.memory_thread.select(column_name).with_columns(
            pl.lit("text-embedding-ada-002").alias("model")
        )
        input_df = df.with_columns(df[column_name].alias("input")).drop(column_name)
        return input_df

    def embed_column(
        self, column: str = "content", generator_log_name: str = "chat_embedding"
    ):
        input_df = self.prepare_column_for_embeddings(column)
        embedder = PolarsGenerator(
            input_df=input_df, name=f"{generator_log_name}_text-embedding-ada-002"
        )
        embedder.execute()
        out_path = (
            f"./batch_generator/{generator_log_name}_text-embedding-ada-002_output.ndjson"
        )
        output = load_generated_content(out_path)
        self.memory_thread = self.memory_thread.with_columns(output)

    def convert_column_to_messages(
        self,
        column_name: str,
        model_name: str = "gpt-3.5-turbo-16k",
        system_prompt: str = "Youre a Helpful Summarizer!",
    ):
        df = self.memory_thread.select(column_name).with_columns(
            pl.lit(model_name).alias("model")
        )

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
        self.memory_thread = self.memory_thread.with_columns(input_df)

    def generate_column(
        self,
        column_name: str,
        generator_log_name: str = "chat_summary",
        model_name: str = "gpt-3.5-turbo-16k",
        system_prompt: str = "Youre a Helpful Summarizer!",
    ):
        # TODO: Generate column with OpenAI functionAPI
        self.convert_column_to_messages(
            column_name=column_name, model_name=model_name, system_prompt=system_prompt
        )
        generator = PolarsGenerator(
            input_df=self.memory_thread, name=generator_log_name
        )
        generator.execute()
        out_path = f"./batch_generator/{generator_log_name}_output.ndjson"
        output = load_generated_content(out_path)
        self.memory_thread = self.memory_thread.with_columns(output)

    def pipeline(
        self,
        name: str,
        summary_model: str = "gpt-3.5-turbo-16k",
        embedding_model: str = "text-embedding-ada-002",
        topic_generation_model: str = "gpt-3.5-turbo-16k",
        run_summary: bool = False,
    ):
        grouped_df = self.memory_thread.groupby("conversation_id").agg(
            pl.col("content").alias("content")
        )
        # convert content column to str
        grouped_df = grouped_df.with_columns(
            pl.col("content")
            .apply(lambda x: " ".join(x), return_dtype=pl.Utf8)
            .alias("concatenated_content")
        )
        # tokenize
        grouped_df = self.tokenize_column(
            input_df=grouped_df, column_name="concatenated_content"
        )
        conv_id_column = grouped_df.select("conversation_id")
        if run_summary:
            # summarize
            out_path = f"./batch_generator/{name}_output.ndjson"
            if os.path.exists(out_path):
                output = load_generated_content(out_path)
                summary_column = output.select("output")
                grouped_df = grouped_df.with_columns(summary_column)
                grouped_df = grouped_df.rename({"output": "summary"})
            else:
                summary_column = SummaryColumnProcessor(
                    grouped_df, name=name, model_name=summary_model
                ).execute()
                grouped_df = grouped_df.with_columns(summary_column)
                grouped_df = grouped_df.rename({"output": "summary"})
            embedable_column = "summary"
        else:
            embedable_column = "concatenated_content"

        # embed Summary
        out_path = f"./batch_generator/{name}_{embedding_model}_output.ndjson"
        if os.path.exists(out_path):
            output = load_generated_content(out_path)
            emb_column = output.select("output")
            grouped_df = grouped_df.with_columns(emb_column)
            grouped_df = grouped_df.rename({"output": "summary_embedding"})
        else:
            emb_column = embed_input_column(
                df=grouped_df,
                name=name,
                input_column=embedable_column,
                model_name=embedding_model,
            )
            grouped_df = grouped_df.with_columns(emb_column)
            grouped_df = grouped_df.rename({"output": "summary_embedding"})
        # cluster
        embeddings = grouped_df["summary_embedding"].to_list()
        new_series = cluster_summaries(embeddings)
        grouped_df = grouped_df.with_columns(new_series)
        grouped_df = grouped_df.with_columns(conv_id_column)
        cluster_df = grouped_df.groupby("cluster|summary").agg(
            pl.col(embedable_column).alias(embedable_column)
        )
        # convert content column to str
        cluster_df = cluster_df.with_columns(
            pl.col(embedable_column)
            .apply(lambda x: " ".join(x), return_dtype=pl.Utf8)
            .alias("concatenated_summary")
        )
        cluster_id_column = cluster_df.select("cluster|summary")
        # tokenize
        cluster_df = self.tokenize_column(
            input_df=cluster_df, column_name="concatenated_summary"
        )
        #return cluster_df

        summary_column = TopicLabelProcessor(
            df=cluster_df, name=name, model_name=topic_generation_model
        ).execute()
        cluster_df = cluster_df.with_columns(summary_column)
        cluster_df = cluster_df.with_columns(cluster_id_column)
        cluster_df = cluster_df.rename({"output": "topic_label"})

        joined_df1 = grouped_df.join(
            cluster_df,
            left_on="cluster|summary",
            right_on="cluster|summary",
            how="inner",
        )
        joined_df2 = self.memory_thread.join(
            joined_df1,
            left_on="conversation_id",
            right_on="conversation_id",
            how="inner",
        )
        self.memory_thread = joined_df2

        return grouped_df, cluster_df, joined_df2

