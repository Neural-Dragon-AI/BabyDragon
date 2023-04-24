import openai

class OpenAiEmbedder:
    def get_embedding_size(self):
        return 1536
    def embed(self, data, embed_mark = False, verbose = False):
        try:
            if embed_mark is False and type(data) is dict and "content" in data:
                if verbose is True:
                    print("Embedding without mark", data["content"])
                out = openai.Embedding.create(input=data["content"], engine='text-embedding-ada-002')
            else:
                if verbose is True:
                    print("Embedding without preprocessing the input", data)
                out = openai.Embedding.create(input=str(data), engine='text-embedding-ada-002')
        except:
            raise ValueError("The data  is not valid", data)
        return out.data[0].embedding