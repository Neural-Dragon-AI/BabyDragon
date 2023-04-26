import cohere as co

COHERE_EMBEDDING_SIZE = 512

class CohereEmbedder:
    def get_embedding_size(self):
        return COHERE_EMBEDDING_SIZE
    def embed(self, data, embed_mark = False, verbose = False):
        try:
            if embed_mark is False and type(data) is dict and "content" in data:
                if verbose is True:
                    print("Embedding without mark", data["content"])
                out = co.embed(input=data["content"]).embeddings
            else:
                if verbose is True:
                    print("Embedding without preprocessing the input", data)
                out = co.embed(input=str(data)).embeddings

        except:
            raise ValueError("The data  is not valid", data)
        return out