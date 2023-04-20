from babydragon.memory.threads.base_thread import BaseThread
from babydragon.memory.indexes.memory_index import MemoryIndex
from babydragon.oai_utils.utils import check_dict, mark_question
import faiss
import numpy as np

class VectorThread(BaseThread, MemoryIndex):
    """ vector BaseThread, creates a faiss index with the messages and allows to search for similar messages, memory BaseThread can return messages in either similarity or chronological order 
      add a parameter to choose the order of the messages
    """
    def __init__(self, index = None, name= 'vector_memory', max_context = 2048, max_memory = None, use_mark = False):
        super().__init__(name= name , max_memory= None)
        MemoryIndex.__init__(self, index = index, name = name)
        self.max_context = max_context
        self.use_mark = use_mark
        self.local_index = faiss.IndexFlatIP(self.embedder.get_embedding_size())
        
    def index_message(self,message_dict: dict, verbose: bool =False):
        """index a message in the faiss index, the message is embedded and added to the index
        self.values and self.embeddings and self.index are updated
        """
        message_dict = check_dict(message_dict)
        self.add_to_index(value = message_dict, verbose = verbose)

    def add_message(self, message_dict: dict, verbose: bool = False):
        """add a message to the memory thread, the message is embedded and added to the index
        self.values and self.embeddings and self.index are updated. If use_mark is False only the content of the messages is embedded
        """
        # print("checking the dict")
        message_dict = check_dict(message_dict)
        # print("trying to add the message")
        super().add_message(message_dict)
        if not self.use_mark:
            message = message_dict["content"]
        else:
            message= message_dict
        self.index_message(message, verbose = verbose)
        return True
    
    def token_bound_query(self, query, k: int =10, max_tokens: int =4000):
        """ returns the k most similar messages to the query, sorted in similarity order"""
        if self.use_mark:
            query = mark_question(query)
        return MemoryIndex.token_bound_query(self, query, k, max_tokens)
    
    def sorted_query(self, query, k: int =10, max_tokens: int =4000, reverse: bool = False, return_from_thread = True):
        """ returns the k most similar messages to the query, sorted in chronological order with the most recent message first
        if return_from_thread is True the messages are returned from the memory thread, otherwise they are returned from the index
        if reverse is True the messages are returned in reverse chronological order, with the oldest message first
        """
        unsorted_messages, unsorted_scores, unsorted_indices = self.token_bound_query(query, k, max_tokens=max_tokens)
        #sort the messages 
        
        sorted_messages = [unsorted_messages[i] for i in sorted(range(len(unsorted_messages)), key=lambda k: unsorted_indices[k])]
        sorted_scores = [unsorted_scores[i] for i in sorted(range(len(unsorted_scores)), key=lambda k: unsorted_indices[k])]
        sorted_indices = [unsorted_indices[i] for i in sorted(range(len(unsorted_indices)), key=lambda k: unsorted_indices[k])]
        if reverse:
            sorted_messages.reverse()
            sorted_scores.reverse()
            sorted_indices.reverse()
        if return_from_thread:
            sorted_messages = [self.memory_thread[i] for i in sorted_indices]
        return sorted_messages, sorted_scores, sorted_indices
    
    def weighted_query(self, query, k: int = 10, max_tokens: int = 4000, history_k: int = 10, decay_factor: float = 0.5):
        sorted_messages, sorted_scores, sorted_indices = self.sorted_query(query, k, max_tokens)
        recent_messages, recent_indices = self.token_bound_history(max_tokens, max_history=history_k)

        combined_messages = sorted_messages + recent_messages
        combined_indices = sorted_indices + recent_indices
        combined_embeddings = np.vstack([self.embeddings[i] for i in sorted_indices] + [self.embed(msg["content"]) for msg in recent_messages])
        self.local_index.add(combined_embeddings)

        _, local_indices = self.local_index.search(self.embed(query).reshape(1, -1), len(self.local_index))

        weights = [np.exp(-decay_factor * i) for i in range(len(combined_indices))]
        weighted_scores = [sorted_scores[i] * weights[i] for i in local_indices[0]]

        sorted_weighted_indices = sorted(range(len(weighted_scores)), key=lambda x: weighted_scores[x], reverse=True)
        weighted_results = [combined_messages[i] for i in sorted_weighted_indices]

        self.local_index.reset()

        return weighted_results
    

