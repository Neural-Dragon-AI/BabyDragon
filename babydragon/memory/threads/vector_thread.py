from typing import Optional, Tuple, List

import faiss
import numpy as np

from babydragon.memory.indexes.memory_index import MemoryIndex
from babydragon.memory.threads.base_thread import BaseThread
from babydragon.utils.chatml import check_dict, mark_question


class VectorThread(BaseThread, MemoryIndex):
    """vector BaseThread, creates a faiss index with the messages and allows to search for similar messages, memory BaseThread can return messages in either similarity or chronological order
    add a parameter to choose the order of the messages
    """

    def __init__(self, name="vector_memory", max_context=2048, use_mark=False):
        BaseThread.__init__(self, name=name, max_memory=None)
        MemoryIndex.__init__(self, index=None, name=name)
        self.max_context = max_context
        self.use_mark = use_mark
        self.local_index = faiss.IndexFlatIP(self.embedder.get_embedding_size())

    def index_message(self, message: str, verbose: bool = False):
        """index a message in the faiss index, the message is embedded and added to the index
        self.values and self.embeddings and self.index are updated
        """

        self.add_to_index(value=message, verbose=verbose)

    def add_message(self, message_dict: dict, verbose: bool = False):
        """add a message to the memory thread, the message is embedded and added to the index
        self.values and self.embeddings and self.index are updated. If use_mark is False only the content of the messages is embedded
        """
        # print("checking the dict")
        message_dict = check_dict(message_dict)
        # print("trying to add the message")
        BaseThread.add_message(self, message_dict)
        # print(message_dict)
        message = message_dict["content"]
        self.index_message(message, verbose=verbose)
        return True

    def token_bound_query(self, query, k: int = 10, max_tokens: int = 4000):
        """returns the k most similar messages to the query, sorted in similarity order"""
        if self.use_mark:
            query = mark_question(query)
        return MemoryIndex.token_bound_query(self, query, k, max_tokens)

    def sorted_query(
        self,
        query,
        k: int = 10,
        max_tokens: int = 4000,
        reverse: bool = False,
        return_from_thread=True,
    ) -> Tuple[List[str], List[float], List[int]]:
        """returns the k most similar messages to the query, sorted in chronological order with the most recent message first
        if return_from_thread is True the messages are returned from the memory thread, otherwise they are returned from the index
        if reverse is True the messages are returned in reverse chronological order, with the oldest message first
        """
        unsorted_messages, unsorted_scores, unsorted_indices = self.token_bound_query(query, k, max_tokens=max_tokens)

        num_results = min(len(unsorted_messages), len(unsorted_scores), len(unsorted_indices))
        # unsorted_indices = [int(i) for i in unsorted_indices]  # convert numpy arrays to integers
        unsorted_indices = [int(i) for sublist in unsorted_indices for i in sublist]

        # Sort the indices
        sorted_indices = sorted(range(num_results), key=lambda x: unsorted_indices[x])
        
        print(sorted_indices)
        print(type(sorted_indices))

        if reverse:
            sorted_indices.reverse()

        # Fetch the sorted messages, scores, and indices based on sorted_indices
        sorted_messages = [unsorted_messages[i] for i in sorted_indices]
        sorted_scores = [unsorted_scores[i] for i in sorted_indices]
        sorted_indices = [unsorted_indices[i] for i in sorted_indices]

        if return_from_thread:
            sorted_messages = [self.memory_thread[i] for i in sorted_indices]

        return sorted_messages, sorted_scores, sorted_indices
    def weighted_query(
        self,
        query,
        k: int = 10,
        max_tokens: int = 4000,
        decay_factor: float = 0.1,
        temporal_weight: float = 0.5,
        order_by: str = "chronological",
        reverse: bool = False,
    ) -> list:
        """Returns the k most similar messages to the query, sorted in either similarity or chronological order. The results are weighted by a combination of similarity scores and temporal weights.
        The temporal weights are computed using an exponential decay function with the decay factor as the decay rate. The temporal weight of the most recent message is 1 and the temporal weight of the oldest message is 0.
        The temporal weight of a message is multiplied by the temporal_weight parameter to control the relative importance of the temporal weights. The default value of 0.5 means that the temporal weights are equally important as the similarity scores.
        The order_by parameter controls the order of the results. If it is set to 'similarity', the results are sorted in similarity order. If it is set to 'chronological', the results are sorted in chronological order with the most recent message first.
        If reverse is True, the results are sorted in reverse chronological order with the oldest message first.
        """
        # Validate order_by parameter
        if order_by not in ("similarity", "chronological"):
            raise ValueError(
                "Invalid value for order_by parameter. It should be either 'similarity' or 'chronological'."
            )

        # Get similarity-based results
        sim_messages, sim_scores, sim_indices = self.sorted_query(
            query, k, max_tokens=max_tokens
        )

        # Get token-bound history
        hist_messages, hist_indices = self.token_bound_history(max_tokens=max_tokens)

        # Combine messages and indices
        combined_messages = sim_messages + hist_messages
        combined_indices = sim_indices + hist_indices

        # Create the local_index and populate it
        self.local_index = MemoryIndex(name="local_index")
        for message in combined_messages:
            self.local_index.add_to_index(value=message, verbose=False)

        # Perform a new query on the combined index
        (
            new_query_results,
            new_query_scores,
            new_query_indices,
        ) = self.local_index.token_bound_query(
            query, k=len(combined_messages), max_tokens=max_tokens
        )

        # Compute temporal weights
        temporal_weights = [
            np.exp(-decay_factor * i) for i in range(len(combined_messages))
        ]
        temporal_weights = [
            w / sum(temporal_weights) for w in temporal_weights
        ]  # Normalize the temporal weights

        # Combine similarity scores and temporal weights
        weighted_scores = []
        for i in range(len(new_query_scores)):
            sim_score = new_query_scores[i]
            temp_weight = temporal_weights[combined_indices.index(new_query_indices[i])]
            weighted_score = (
                1 - temporal_weight
            ) * sim_score + temporal_weight * temp_weight
            weighted_scores.append(weighted_score)

        # Sort the results based on the order_by parameter
        if order_by == "similarity":
            sorting_key = lambda k: weighted_scores[k]
        elif order_by == "chronological":  # order_by == 'chronological'
            sorting_key = lambda k: new_query_indices[k]
        else:
            raise ValueError(
                "Invalid value for order_by parameter. It should be either 'similarity' or 'chronological'."
            )

        sorted_indices = [
            new_query_indices[i]
            for i in sorted(
                range(len(new_query_indices)), key=sorting_key, reverse=not reverse
            )
        ]
        sorted_results = [
            new_query_results[i]
            for i in sorted(
                range(len(new_query_results)), key=sorting_key, reverse=not reverse
            )
        ]
        sorted_scores = [
            weighted_scores[i]
            for i in sorted(
                range(len(weighted_scores)), key=sorting_key, reverse=not reverse
            )
        ]

        # Return only the top k results without exceeding max_tokens
        final_results, final_scores, final_indices = [], [], []
        current_tokens = 0
        for i in range(min(k, len(sorted_results))):
            message_tokens = self.get_message_tokens(sorted_results[i])
            if current_tokens + message_tokens <= max_tokens:
                final_results.append(sorted_results[i])
                final_scores.append(sorted_scores[i])
                final_indices.append(sorted_indices[i])
                current_tokens += message_tokens
            else:
                break

        return final_results, final_scores, final_indices
