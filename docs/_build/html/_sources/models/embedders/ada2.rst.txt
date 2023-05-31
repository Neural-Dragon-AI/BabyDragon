ada2
====

.. code-block:: python

	
	class OpenAiEmbedder:
	    def get_embedding_size(self):
	        return ADA_EMBEDDING_SIZE
	
	    def embed(self, data, verbose=False):
	
	        if type(data) is dict and "content" in data:
	            if verbose is True:
	                print("Embedding without mark", data["content"])
	            out = openai.Embedding.create(
	                input=data["content"], engine="text-embedding-ada-002"
	            )
	        else:
	            if len(tokenizer.encode(data)) > MAX_CONTEXT_LENGTH:
	                raise ValueError(f" The input is too long for OpenAI, num tokens is {len(tokenizer.encode(data))}, instead of {MAX_CONTEXT_LENGTH}")
	            if verbose is True:
	                print("Embedding without preprocessing the input", data)
	            out = openai.Embedding.create(
	                input=str(data), engine="text-embedding-ada-002"
	            )
	        return out.data[0].embedding
	
	    def batch_embed(self, data: List[str]):
	        if type(data) is dict and "content" in data:
	            raise ValueError("Batch embedding not supported for dictionaries")
	        elif type(data) is str:
	            return self.embed(data)
	        elif type(data) is list:
	            out = openai.Embedding.create(
	                input=data, engine="text-embedding-ada-002"
	            )
	            embeddings = []
	            for embedding in out.data:
	                embeddings.append(embedding.embedding)
	            return embeddings
	

.. automodule:: ada2
   :members:
