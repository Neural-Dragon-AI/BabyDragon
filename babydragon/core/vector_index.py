import faiss
import numpy as np
import pickle
from IPython.display import display, Markdown
import pandas as pd
import copy
import os
from babydragon.oai_utils.utils import OpenAiEmbedder
import tiktoken 
import time

class MemoryIndex:
    """ this class is a wrapper for a faiss index, it contains information about the format of the index the faiss index itself"""
    def __init__(self, index = None,values = None, embeddings = None, name='memory_index', save_path = None, load= False,):
        self.name = name
        self.embedder = OpenAiEmbedder()
        self.save_path = save_path
        # with load been through we search for a pickle file with the same name of the index
        if load is True:
            self.load()
        else:
            self.init_index(index,values,embeddings)
        self.tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')
        self.hints_list = []


    def init_index(self,index,values,embeddings):
        #fist case is when we create a new index from scratch
        if index is None and values is None and embeddings is None :
            print("Creating a new index")
            self.index = faiss.IndexFlatIP(self.embedder.get_embedding_size())
            self.values = []
        #second case is where we create the index from a list of embeddings
        elif index is None and values is not None and embeddings is not None and len(values) == len(embeddings):
            print("Creating a new index from a list of embeddings and values")
            self.index = faiss.IndexFlatIP(self.embedder.get_embedding_size())
            for embedding,value in zip(embeddings,values):
                self.add_to_index_embedding(value, embedding) 
        #third case is where we create the index from a faiss index and values list  
        elif isinstance(index, faiss.Index) and index.d == self.embedder.get_embedding_size() and type(values) == list and len(values) == index.ntotal:
            print("Creating a new index from a faiss index and values list")
            self.index = index
            self.values = values
        #fourth case is where we create an index from a list of values, the values are embedded and the index is created
        elif index is None and values is not None and embeddings is None:
            print("Creating a new index from a list of values")
            self.index = faiss.IndexFlatIP(self.embedder.get_embedding_size())
            i = 0
            for value in values:
                #print the value id to see the progress
                print("Embedding value ", i, " of ", len(values))
                #start tracking the time using time
                start = time.time()
                self.add_to_index(value)
                #print the time it took to embed the value
                print("Embedding value ", i, " took ", time.time() - start, " seconds")
                i+=1
        else:
            raise ValueError("The index is not a valid faiss index or the embedding dimension is not correct")

    def add_to_index(self,value, verbose = True, steps=0):
        """index a message in the faiss index, the message is embedded and the id is saved in the values list
        """
        if value not in self.values:
            # try:
            embedding = self.embedder.embed(value)
            if verbose:
                display(Markdown("The value {value} was embedded".format(value = value))) 
            # except:
            #     #if user stops the embedding process we stop the embedding process
            #     if KeyboardInterrupt:
            #         raise KeyboardInterrupt

            #     if verbose:
            #         display(Markdown("The value {value} was not embedded, trying again".format(value = value)))
            #     steps+=1
            #     if steps < 5:
            #         self.add_to_index(value, verbose, steps)
            #     else:
            #         display(Markdown("The value {value} was not embedded, giving up".format(value = value)))                

            self.index.add(np.array([embedding]).astype(np.float32))
            self.values.append(value)
        else:
            display(Markdown("The value {value} was already in the index".format(value = value)))

    def add_to_index_embedding(self, value, embedding, verbose = False):
        """index a message in the faiss index, the message is embedded and the id is saved in the values list
        """
        #check that the embedding is of the correct size and type, the type can be
        # list of floats, numpy array of floats, string of a list of floats
        # if list of floats convert to numpy array 
        # if string convert to list of floats using eval and then to numpy array
        if type(embedding) is list:
            embedding = np.array([embedding])
        elif type(embedding) is str:
            embedding = eval(embedding)
            embedding = np.array([embedding]).astype(np.float32)
        elif type(embedding) is not np.ndarray:
            raise ValueError("The embedding is not a valid type")
        if  value not in self.values:
            self.index.add(embedding)
            self.values.append(value)
        else:
            if verbose:
                display(Markdown("The value {value} was already in the index".format(value = value)))

    def get_embedding_by_index(self, index):
        """
        Get the embedding corresponding to a certain index value.

        Args:
            index (int): The index of the value for which the embedding is required.

        Returns:
            numpy.ndarray: The embedding corresponding to the index value.
        """
        if index < 0 or index >= len(self.values):
            raise ValueError("The index is out of range")

        # Fetch the embedding from the Faiss index
        embedding = self.index.reconstruct(index)

        return embedding
    
    def get_index_by_value(self, value):
        """
        Get the index corresponding to a value in self.values.

        Args:
            value (str): The value for which the index is required.

        Returns:
            int: The index corresponding to the value, or None if the value is not in self.values.
        """
        if value in self.values:
            index = self.values.index(value)
            return index
        else:
            return None

    def get_embedding_by_value(self, value):
        """
        Get the embedding corresponding to a certain value in self.values.

        Args:
            value (str): The value for which the embedding is required.

        Returns:
            numpy.ndarray: The embedding corresponding to the value, or None if the value is not in self.values.
        """
        index = self.get_index_by_value(value)
        if index is not None:
            embedding = self.get_embedding_by_index(index)
            return embedding
        else:
            return None
    def faiss_query(self, key, k = 10):
        # Embed the data
        embedding = self.embedder.embed(key)
        if k > len(self.values):
            k = len(self.values)
        # Query the Faiss index for the top-K most similar values
        D, I = self.index.search(np.array([embedding]).astype(np.float32), k)
        values = [self.values[i] for i in I[0]]
            
        return values
    def save(self, path=None):
        """saves the index and values to a pickle file"""
        if path is None and self.save_path is None:
            path = self.name + ".pkl"
        elif path is None and self.save_path is not None:
            if self.save_path.endswith("/"):
                path = self.save_path + self.name + ".pkl"
            else:
                path = self.save_path + "/" + self.name + ".pkl"
        print("Saving the index to ", path)
        with open(path, 'wb') as f:
            pickle.dump({'index': self.index, 'values': self.values}, f)

    def load(self, path=None):
        """loads the index and values from a pickle file"""
        if path is None and self.save_path is None:
            path = self.name + ".pkl"
        elif path is None and self.save_path is not None:
            if self.save_path.endswith("/"):
                path = self.save_path + self.name + ".pkl"
            else:
                path = self.save_path + "/" + self.name + ".pkl"

        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.index = data['index']
            self.values = data['values']

    def get_token_bound_hints(self, query, k = 10, max_context = 4000):
            context_tokens = 0
            if len(self.values) > 0 :
                top_k = self.faiss_query(query, k = min(k, len(self.values)))
                # print("top_k: ", top_k)
                top_k_hint = []
                for hint in top_k:
                    #mark the message and gets the length in tokens
                    message_tokens = len(self.tokenizer.encode(hint))
                    if context_tokens+message_tokens <= max_context:
                        top_k_hint+=[hint]
                        context_tokens += message_tokens
                #inver the top_k_prompt to start from the most similar message
                # top_k_hint.reverse()
                #reverse the prompt so that last is the most similar message
                self.hints_list.append({"query": query, "hints": top_k_hint})
            return top_k_hint



