from typing import List
from babydragon.utils.main_logger import logger
import libcst as cst
import numpy as np
import openai
import tiktoken

import time



ADA_EMBEDDING_SIZE = 1536
MAX_CONTEXT_LENGTH = 8100

TOKENIZER = tiktoken.encoding_for_model("gpt-3.5-turbo")

class OpenAiEmbedder:

    def get_embedding_size(self):
        return ADA_EMBEDDING_SIZE

    def embed(self, data, verbose=False):
        if isinstance(data, list) and len(data) > 1:
            logger.info("Batch embedding")
            return self.batch_embed(data)
        elif isinstance(data, list) and len(data) == 1:
            logger.info("Serial embedding")
            data = data[0]

        if isinstance(data, dict) and "content" in data:
            if verbose:
                logger.info("Embedding without mark", data["content"])
            out = openai.Embedding.create(
                input=data["content"], engine="text-embedding-ada-002"
            )
        else:
            if len(TOKENIZER.encode(data)) > MAX_CONTEXT_LENGTH:
                raise ValueError(f" The input is too long for OpenAI, num tokens is {len(TOKENIZER.encode(data))}, instead of {MAX_CONTEXT_LENGTH}")
            if verbose:
                logger.info(f"Embedding without preprocessing the input {data}")
            out = openai.Embedding.create(
                input=str(data), engine="text-embedding-ada-002"
            )
        return out.data[0].embedding

    def batch_embed(self, data: List[str], batch_size: int = 1000):
        if isinstance(data, dict) and "content" in data:
            raise ValueError("Batch embedding not supported for dictionaries")
        elif isinstance(data, str):
            raise ValueError("Batch embedding not supported for strings use embed() instead")
        elif isinstance(data, list):
            batch = []
            embeddings = []
            i = 1
            total_number_of_batches = len(data)//batch_size + 1 if len(data) % batch_size > 0 else len(data)//batch_size
            for value in data:
                batch.append(value)
                if len(batch) == batch_size:
                    start = time.time()
                    out = openai.Embedding.create(
                        input=batch, engine="text-embedding-ada-002"
                    )
                    for embedding in out.data:
                        embeddings.append(embedding.embedding)
                    logger.info(f"Batch {i} of {total_number_of_batches}")
                    logger.info(f"Embedding batch {i} took {time.time() - start} seconds")
                    i += 1
                    batch = []
            if len(batch) > 0:
                start = time.time()
                out = openai.Embedding.create(
                    input=batch, engine="text-embedding-ada-002"
                )
                for embedding in out.data:
                    embeddings.append(embedding.embedding)
                logger.info(f"Batch {i} of {total_number_of_batches}")
                logger.info(f"Embedding batch {i} took {time.time() - start} seconds")
            logger.info(f'Total number of embeddings {len(embeddings)}')

            if len(embeddings) != len(data):
                raise ValueError("The number of embeddings is different from the number of values an error occured in OpenAI API during the embedding process")
            return embeddings



def parse_and_embed_functions(input_str: str) -> List[np.ndarray]:
    # Parse the input string with libcst
    module = cst.parse_module(input_str)

    # Find all the functions in the module and embed them separately
    embeddings = []
    for node in module.body:

        if isinstance(node, cst.FunctionDef) or isinstance(node, cst.ClassDef):
            func_str = cst.Module(body=[node]).code
            print("Function string", func_str)
            embedding = openai.Embedding.create(
                input=str(func_str)[:MAX_CONTEXT_LENGTH],
                engine="text-embedding-ada-002",
            )
            if embedding is not None:
                embeddings.append(embedding.data[0].embedding)

    avg_embedding = avg_embeddings(embeddings)
    print(avg_embedding.shape)
    return avg_embedding


def avg_embeddings(embeddings: List[np.ndarray]) -> np.ndarray:
    print("Embeddings len", len(embeddings))
    # convert embeddings to numpy array
    embeddings = np.array(embeddings)
    print("Embedding Matrix Shape", embeddings.shape)
    return np.array([np.sum(embeddings.T, axis=1)]).astype(np.float32)
