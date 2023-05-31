cohere
======

.. code-block:: python

	
	
	class CohereEmbedder:
	    def get_embedding_size(self):
	        return COHERE_EMBEDDING_SIZE
	
	    def embed(self, data, verbose=False):
	        if type(data) is dict and "content" in data:
	            if verbose is True:
	                print("Embedding from dictionary", data["content"])
	                response = co.embed(texts= data["content"],model='multilingual-22-12')
	        else:
	            if len(tokenizer.encode(data)) > MAX_CONTEXT_LENGTH:
	                raise ValueError(f" The input is too long for Cohere, num tokens is {len(tokenizer.encode(data))}, instead of {MAX_CONTEXT_LENGTH}")
	            if verbose is True:
	                print("Embedding without preprocessing the input", data)
	            response = co.embed(texts=[str(data)],model='multilingual-22-12')
	        return response.embeddings[0]
	

.. automodule:: cohere
   :members:
