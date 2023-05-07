import cohere
import tiktoken
co = cohere.Client('LeohkffIg5ucAxbMSfiCIhZ0RL9M2uuw0GVb99ZN')
COHERE_EMBEDDING_SIZE = 512
MAX_CONTEXT_LENGTH = 512
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")


class CohereEmbedder:
    def get_embedding_size(self):
        return COHERE_EMBEDDING_SIZE

    def embed(self, data, verbose=False):
        if type(data) is dict and "content" in data:
            if verbose is True:
                print("Embedding from dictionary", data["content"])
                response = co.embed(texts= data["content"],model='small')
        else:
            if len(tokenizer.encode(data)) > MAX_CONTEXT_LENGTH:
                raise ValueError(f" The input is too long for Cohere, num tokens is {len(tokenizer.encode(data))}, instead of {MAX_CONTEXT_LENGTH}")
            if verbose is True:
                print("Embedding without preprocessing the input", data)
            response = co.embed(texts=[str(data)],model='small')
        return response.embeddings[0]

