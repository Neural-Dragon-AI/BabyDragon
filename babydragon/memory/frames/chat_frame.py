from typing import Any, Optional, List, Union, Dict
from babydragon.memory.threads.base_thread import BaseThread
from babydragon.models.embedders.ada2 import OpenAiEmbedder
from babydragon.models.embedders.cohere import CohereEmbedder
from babydragon.models.generators.PolarsGenerator import PolarsGenerator
import json
import polars as pl
import numpy as np
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class ChatFrame(BaseThread):
    def __init__(self, name: str = "chat_frame",
                 context_columns: List[str] = [],
                 embeddable_columns: List[str] = ['content'],
                 embedding_columns: List[str] = [],
                 text_embedder: Optional[Union[OpenAiEmbedder, CohereEmbedder]] = OpenAiEmbedder,
                 markdown: str = "text/markdown",
                 max_memory: int | None = None,
                 tokenizer: Any | None = None,
                 save_path: str = 'threads') -> None:

        BaseThread.__init__(self, name, max_memory, tokenizer, save_path)
        self.context_columns = context_columns
        self.embeddable_columns = embeddable_columns
        self.embedding_columns = embedding_columns
        self.text_embedder = text_embedder
        self.markdown = markdown


    # Dot Product Query
    def search_column_with_dot_product(self, query: str, embeddable_column_name: str, top_k: int) -> pl.DataFrame:
        embedding_column_name = 'embedding|' + embeddable_column_name

        query_as_series = pl.Series(query)
        dot_product_frame = self.memory_thread.with_columns(self.memory_thread[embedding_column_name].list.eval(pl.element().explode().dot(query_as_series),parallel=True).list.first().alias("dot_product"))
        # Sort by dot product and select top_k rows
        result = dot_product_frame.sort('dot_product', descending=True).slice(0, top_k)
        return result

    # Tokenization
    def tokenize_column(self, column_name: str):
        new_values = self.tokenizer.encode_batch(self.memory_thread[column_name].to_list())
        new_series = pl.Series(f'tokens|{column_name}', new_values)
        len_values = [len(x) for x in new_values]
        new_series_len = pl.Series(f'tokens_len|{column_name}', len_values)
        self.memory_thread = self.memory_thread.with_columns(new_series)
        self.memory_thread = self.memory_thread.with_columns(new_series_len)
    
    def prepare_column_for_embeddings(self, column_name):

        df = self.memory_thread.select(column_name).with_columns(pl.lit("text-embedding-ada-002").alias("model"))
        input_df = df.with_columns(df[column_name].alias('input')).drop(column_name)

        return input_df
    
    def embed_column(self, column, generator_log_name="chat_embedding"):
        input_df = self.prepare_column_for_embeddings(column)
        embedder = PolarsGenerator(input_df = input_df, name = f"{generator_log_name}_text-embedding-ada-002")
        embedder.execute()
        out_path = f"./batch_generator/{generator_log_name}_text-embedding-ada-002.ndjson"
        #load output file to list
        with open(out_path) as f:
            output = f.readlines()
        #add to memory
        output = [x.strip() for x in output]
        output = [json.loads(x) for x in output]
        #reverse order
        output = output[::-1]
        output = pl.DataFrame(output)
        self.memory_thread = self.memory_thread.with_columns(output)


    def convert_column_to_messages(self, column_name, model_name = "gpt-3.5-turbo-16k", system_prompt = "Youre a Helpful Summarizer!"):
        df = self.memory_thread.select(column_name).with_columns(pl.lit(model_name).alias("model"))

        def create_content(value):
            return ([{"role": "system", "content":system_prompt},
                        {"role": "user", "content": f"{value}"}])

        input_df = df.with_columns(df[column_name].apply(create_content, return_dtype=pl.List).alias('messages')).drop(column_name)
        self.memory_thread = self.memory_thread.with_columns(input_df)

    def generate_column(self, column_name, generator_log_name="chat_summary",  model_name = "gpt-3.5-turbo-16k", system_prompt = "Youre a Helpful Summarizer!"):
        #TODO: Generate column with OpenAI functionAPI
        self.convert_column_to_messages(column_name = column_name, model_name = model_name, system_prompt = system_prompt)
        generator = PolarsGenerator( input_df = self.memory_thread, name = generator_log_name)
        generator.execute()
        out_path = f"./batch_generator/{generator_log_name}_output.ndjson"
        #load output file to list
        with open(out_path) as f:
            output = f.readlines()
        #add to memory
        output = [x.strip() for x in output]
        output = [json.loads(x) for x in output]
        #reverse order
        output = output[::-1]
        output = pl.DataFrame(output)
        self.memory_thread = self.memory_thread.with_columns(output)
    
    def reduce_dimensionality(self):
        umap_model = UMAP(n_neighbors=5, n_components=5, min_dist=0.0, metric='cosine')
        embeddings = np.stack(self.memory_thread['embedding|content'].to_list())
        reduced_embeddings = umap_model.fit_transform(embeddings)
        self.memory_thread = self.memory_thread.with_columns(pl.Series('reduced_embedding', reduced_embeddings.tolist()))
    
    def cluster_data(self):
        hdbscan_model = HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
        reduced_embeddings = np.stack(self.memory_thread['reduced_embedding'].to_list())
        cluster_labels = hdbscan_model.fit_predict(reduced_embeddings)
        self.memory_thread = self.memory_thread.with_columns(pl.Series('cluster_labels', cluster_labels))
    
    def vectorize_topics(self):
        vectorizer_model = CountVectorizer(stop_words="english")
        # Concatenate texts within each cluster and then vectorize
        df = self.memory_thread.sort('cluster_labels')
        # Step 2: Initialize an empty list and DataFrame to hold the results
        concatenated_contents = []
        unique_labels = []

        # Step 3: Iterate through unique 'cluster_labels' to perform the custom aggregation
        for label in df["cluster_labels"].unique():
            subset = df.filter(df["cluster_labels"] == label)
            concatenated_content = " ".join(subset["content"].to_list())
            
            concatenated_contents.append(concatenated_content)
            unique_labels.append(label)

        # Step 4: Create the output DataFrame
        final_df = pl.DataFrame({
            "cluster_labels": unique_labels,
            "concatenated_content": concatenated_contents
        })
        concatenated_texts = final_df["concatenated_content"].to_list()
        self.token_matrix = vectorizer_model.fit_transform(concatenated_texts)
    
    def create_topic_representation(self):
        tfidf_transformer = TfidfTransformer()
        self.topic_representations = tfidf_transformer.fit_transform(self.token_matrix)
    
    def execute_tagging_pipeline(self):
        self.reduce_dimensionality()
        self.cluster_data()
        self.vectorize_topics()
        self.create_topic_representation()






