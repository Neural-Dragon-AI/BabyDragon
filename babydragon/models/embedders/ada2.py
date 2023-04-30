import openai
import libcst as cst
from typing import List
import numpy as np
ADA_EMBEDDING_SIZE = 1536
MAX_CONTEXT_LENGTH = 8100

class OpenAiEmbedder:
    def get_embedding_size(self):
        return ADA_EMBEDDING_SIZE
    def embed(self, data, embed_mark = False, verbose = False):

        if embed_mark is False and type(data) is dict and "content" in data:
            if verbose is True:
                print("Embedding without mark", data["content"])
            out = openai.Embedding.create(input=data["content"], engine='text-embedding-ada-002')
        else:
            if len(str(data)) > MAX_CONTEXT_LENGTH:
                data = str(data)[:MAX_CONTEXT_LENGTH]
            if verbose is True:
                print("Embedding without preprocessing the input", data)
            out = openai.Embedding.create(input=str(data), engine='text-embedding-ada-002')
        return out.data[0].embedding

def parse_and_embed_functions(input_str: str) -> List[np.ndarray]:
    # Parse the input string with libcst
    module = cst.parse_module(input_str)

    # Find all the functions in the module and embed them separately
    embeddings = []
    for node in module.body:

        if isinstance(node, cst.FunctionDef) or isinstance(node, cst.ClassDef):
            func_str = cst.Module(body=[node]).code
            print("Function string", func_str)
            embedding = openai.Embedding.create(input=str(func_str)[:MAX_CONTEXT_LENGTH], engine='text-embedding-ada-002')
            if embedding is not None:
                embeddings.append(embedding.data[0].embedding)

    avg_embedding = avg_embeddings(embeddings)
    print(avg_embedding.shape)
    return avg_embedding

def avg_embeddings(embeddings: List[np.ndarray]) -> np.ndarray:
    print("Embeddings len", len(embeddings))
    #convert embeddings to numpy array
    embeddings = np.array(embeddings)
    print("Embedding Matrix Shape", embeddings.shape)
    return np.array([np.sum(embeddings.T, axis=1)]).astype(np.float32)
