sbert
=====

.. code-block:: python

	
	class SBERTEmbedder:
	    def get_embedding_size(self):
	        return SBERT_EMBEDDING_SIZE
	
	    def embed(self,
	        data,
	        key="content",
	        model_name="all-MiniLM-L6-v2",
	        batch_size=128,
	    ):
	        """
	        Embed the sentences/text using the MiniLM language model (which uses mean pooling)
	        """
	        print("Embedding data")
	        model = SentenceTransformer(model_name)
	        print("Model loaded")
	        if isinstance(data, dict):
	            sentences = data[key].tolist()
	            unique_sentences = data[key].unique()
	        elif isinstance(data, str):
	            #breal the string into sentences based on . or ? or !
	            sentences = re.split('[.!?]', data)
	            sentences = [s.strip() for s in sentences if s.strip()]  #
	            #filter empty sentences
	            sentences = list(filter(lambda x: len(x) > 0, sentences))
	            unique_sentences = list(set(sentences))
	        else:
	            raise ValueError(f"Data must be a dictionary with attribute {key} or a string, but got {type(data)} instead")
	        
	        print("Unique sentences", len(unique_sentences))
	        self.unique_sentences = unique_sentences
	        for sentence in unique_sentences:
	            tokens = tokenizer.encode(sentence)
	            if len(tokens) > MAX_CONTEXT_LENGTH:
	                raise ValueError(f" The input subsentence is too long for SBERT, num tokens is {len(tokens)}, instead of {MAX_CONTEXT_LENGTH}")
	
	       
	        embeddings = model.encode(
	                unique_sentences, show_progress_bar=True, batch_size=batch_size
	            )
	
	        print("Embeddings computed")
	
	        mapping = {
	            sentence: embedding
	            for sentence, embedding in zip(unique_sentences, embeddings)
	        }
	        embeddings = np.array([mapping[sentence] for sentence in sentences])
	
	        return np.mean(embeddings, axis=0).tolist()
	

.. automodule:: sbert
   :members:
