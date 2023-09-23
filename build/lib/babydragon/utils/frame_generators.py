import json
from abc import ABC, abstractmethod

import numpy as np
import polars as pl
from hdbscan import HDBSCAN
from umap import UMAP

from babydragon.models.generators.PolarsGenerator import PolarsGenerator


def embed_input_column(
    df, name, input_column="summary", model_name="text-embedding-ada-002"
):
    df = df.select(input_column).with_columns(pl.lit(model_name).alias("model"))
    input_df = df.with_columns(df[input_column].alias("input")).drop(input_column)
    embedder = PolarsGenerator(input_df=input_df, name=f"{name}_{model_name}")
    embedder.execute()
    out_path = f"./batch_generator/{name}_{model_name}_output.ndjson"
    # load output file to list
    output = load_generated_content(out_path)

    emb_column = output.select("output")
    return emb_column


class PolarsProcessor(ABC):
    def __init__(self, df, model_name, input_col, system_prompt=None):
        self.df = df
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.input_col = input_col

    def create_content(self, value, system_prompt):
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": value},
        ]

    def process(self):
        self.df = self.df.select(self.input_col).with_columns(
            pl.lit(self.model_name).alias("model")
        )
        self.df = self.df.with_columns(
            self.df[self.input_col]
            .apply(
                lambda x: self.create_content(x, self.system_prompt),
                return_dtype=pl.List,
            )
            .alias("messages")
        ).drop(self.input_col)

    @abstractmethod
    def execute(self):
        pass


class SummaryColumnProcessor(PolarsProcessor):
    def __init__(
        self,
        df,
        name,
        column_name="concatenated_content",
        model_name="gpt-3.5-turbo-16k",
        system_prompt="You are tasked as a Conversation Summarizer. Please provide a succinct summary of the conversation message below.",
    ):
        super().__init__(df, model_name, column_name, system_prompt)
        self.name = name

    def execute(self):
        self.process()
        generator = PolarsGenerator(input_df=self.df, name=self.name)
        generator.execute()
        out_path = f"./batch_generator/{self.name}_output.ndjson"
        output = load_generated_content(out_path)
        summary_column = output.select("output")
        return summary_column


class TopicLabelProcessor(PolarsProcessor):
    def __init__(
        self,
        df,
        name,
        column_name="concatenated_summary",
        model_name="gpt-3.5-turbo-16k",
        system_prompt="You are tasked with generating topic labels. Provide a topic label that is concise (max 3 words), yet comprehensively captures the essence of the cluster's subject matter. \n Cluster content:\n",
    ):
        super().__init__(df, model_name, column_name, system_prompt)
        self.name = name

    def execute(self):
        self.process()
        topic_batch_id = f"topic_batch_{np.random.randint(0, 1e6)}"
        generator = PolarsGenerator(
            input_df=self.df, name=f"{self.name}_{topic_batch_id}"
        )
        generator.execute()
        out_path = f"./batch_generator/{self.name}_{topic_batch_id}_output.ndjson"
        # load output file to list
        output = load_generated_content(out_path)
        summary_column = output.select("output")
        return summary_column


def cluster_summaries(embeddings):
    dim_reduction_model = UMAP(
        n_neighbors=2, n_components=5, min_dist=0.0, metric="cosine"
    )
    cluster_model = HDBSCAN(
        min_cluster_size=2,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    reduced_embeddings = dim_reduction_model.fit_transform(embeddings)
    labels = cluster_model.fit_predict(reduced_embeddings)
    new_column_name = "cluster|summary"
    new_series = pl.Series(new_column_name, labels)
    return new_series


def load_generated_content(out_path):
    with open(out_path) as f:
        output = f.readlines()
    # add to memory
    output = [x.strip() for x in output]
    output = [json.loads(x) for x in output]
    output = pl.DataFrame(output)
    # sort by id
    output = output.sort("id")
    return output
