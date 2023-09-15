from babydragon.models.generators.PolarsGenerator import PolarsGenerator
import json
import polars as pl

import numpy as np
from umap import UMAP
from hdbscan import HDBSCAN
def generate_summary_column(df, name):
    df = df.select("concatenated_content").with_columns(pl.lit("gpt-3.5-turbo-16k").alias("model"))
    def create_content(value):
        system_prompt = "You are tasked as a Conversation Summarizer. Please provide a succinct summary of the conversation message below."
        return ([{"role": "system", "content":system_prompt},
                    {"role": "user", "content": f"{value}"}])

    input_df = df.with_columns(df["concatenated_content"].apply(create_content, return_dtype=pl.List).alias('messages')).drop("concatenated_content")
    generator = PolarsGenerator( input_df = input_df, name = name)
    generator.execute()
    out_path = f"./batch_generator/{name}_output.ndjson"
    output = load_generated_content(out_path)
    summary_column = output.select("output")
    return summary_column

def embed_summary_column(df, name):
    df = df.select("summary").with_columns(pl.lit("text-embedding-ada-002").alias("model"))
    input_df = df.with_columns(df["summary"].alias('input')).drop("summary")
    embedder = PolarsGenerator(input_df = input_df, name = f"{name}_text-embedding-ada-002")
    embedder.execute()
    out_path = f"./batch_generator/{name}_text-embedding-ada-002_output.ndjson"
    #load output file to list
    output = load_generated_content(out_path)

    emb_column = output.select("output")
    return emb_column

def generate_topic_label_column(df, name):
    df = df.select("concatenated_summary").with_columns(pl.lit("gpt-4").alias("model"))
    def create_content(value):
        system_prompt = "You are tasked with generating topic labels. provide a topic label that is concise max 3 words, yet comprehensively captures the essence of the cluster's subject matter. Output only the topic label—no additional text. if the topic is not cohesive or does nto make sense label it as 'other'."
        return ([{"role": "system", "content":system_prompt},
                    {"role": "user", "content": f"{value}"}])

    input_df = df.with_columns(df["concatenated_summary"].apply(create_content, return_dtype=pl.List).alias('messages')).drop("concatenated_summary")
    topic_batch_id = f"topic_batch_{str(np.random.randint(0,100000))}"
    generator = PolarsGenerator( input_df = input_df, name = f'{name}_{topic_batch_id}')
    generator.execute()
    
    out_path = f"./batch_generator/{name}_{topic_batch_id}_output.ndjson"
    #load output file to list
    output = load_generated_content(out_path)
    summary_column = output.select("output")
    return summary_column

def cluster_summaries(embeddings):
    dim_reduction_model = UMAP(n_neighbors=2, n_components=5, min_dist=0.0, metric='cosine')
    cluster_model = HDBSCAN(min_cluster_size=2, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    reduced_embeddings = dim_reduction_model.fit_transform(embeddings)
    labels = cluster_model.fit_predict(reduced_embeddings)
    new_column_name = f'cluster|summary'
    new_series = pl.Series(new_column_name, labels)
    return new_series


def load_generated_content(out_path):
    with open(out_path) as f:
            output = f.readlines()
    #add to memory
    output = [x.strip() for x in output]
    output = [json.loads(x) for x in output]
    output = pl.DataFrame(output)
    #sort by id
    output = output.sort("id")
    return output