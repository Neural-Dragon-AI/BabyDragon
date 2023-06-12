from typing import Dict, List, Optional, Tuple, Union, Generator

from babydragon.chat.chat import BaseChat, Chat, Prompter
from babydragon.memory.indexes.memory_index import MemoryIndex
from babydragon.memory.indexes.pandas_index import PandasIndex
from babydragon.memory.threads.base_thread import BaseThread
from babydragon.memory.threads.fifo_thread import FifoThread
from babydragon.memory.threads.vector_thread import VectorThread
from babydragon.chat.context_manager import ContextManager
from babydragon.utils.chatml import mark_answer, mark_question, mark_system, apply_threshold
import logging
import numpy as np
from sklearn.metrics.pairwise import  cosine_similarity
import random
from babydragon.models.embedders.ada2 import OpenAiEmbedder


# Add this line at the beginning of your code to configure logging
logging.basicConfig(level=logging.INFO)
EMBEDDER = OpenAiEmbedder()
class FifoChat(FifoThread, Chat):
    """
    A chatbot class that combines FIFO Memory Thread, BaseChat, and Prompter. The oldest messages are removed first
    when reaching the max_memory limit. The memory is defined in terms of tokens, and outs are passed to the
    longterm_memory. The lucid_memory is a redundant memory that stores all the messages.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        index_dict: Optional[Dict[str, Union[PandasIndex, MemoryIndex]]] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        name: str = "fifo_memory",
        max_index_memory: int = 400,
        max_fifo_memory: int = 2048,
        max_output_tokens: int = 1000,
        longterm_thread: Optional[BaseThread] = None,
    ):

        FifoThread.__init__(
            self, name=name, max_memory=max_fifo_memory, longterm_thread=longterm_thread
        )
        Chat.__init__(
            self,
            model=model,
            index_dict=index_dict,
            max_output_tokens=max_output_tokens,
            max_index_memory=max_index_memory,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            name=name,
        )

        self.prompt_func = self.fifo_memory_prompt

    def fifo_memory_prompt(self, message: str) -> Tuple[List[dict], dict]:
        """
        Compose the prompt for the chat-gpt API, including the system prompt and memory thread.

        :param message: A string representing the user message.
        :return: A tuple containing a list of strings as the prompt and the marked question.
        """
        marked_question = mark_question(self.user_prompt(message))
        prompt = (
            [mark_system(self.system_prompt)] + self.memory_thread + [marked_question]
        )
        return prompt, marked_question

    def query(self, question: str, verbose: bool = True, stream: bool = False) -> Union[Generator,str]:
        """
        Query the chatbot with a given question. The question is added to the memory, and the answer is returned
        and added to the memory.

        :param question: A string representing the user question.
        :param verbose: A boolean indicating whether to display input and output messages as Markdown.
        :return: A string representing the chatbot's response.
        """
        marked_question = mark_question(question)
        self.add_message(marked_question)
        answer = BaseChat.query(self, message=question, verbose=verbose, stream=stream)
        if stream:
            return answer
        else:
            self.add_message(answer)
            return answer


class VectorChat(VectorThread, Chat):
    """
    A chatbot class that combines Vector Memory Thread, BaseChat, and Prompter. Memory prompt is constructed by
    filling the memory with the k most similar messages to the question until the max prompt memory tokens are reached.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        index_dict: Optional[Dict[str, Union[PandasIndex, MemoryIndex]]] = None,
        name: str = "vector_memory",
        max_index_memory: int = 400,
        max_vector_memory: int = 2048,
        max_output_tokens: int = 1000,
        system_prompt: str = None,
        user_prompt: str = None,
    
    ):
        VectorThread.__init__(self, name=name, max_context=max_vector_memory)
        Chat.__init__(
            self,
            model=model,
            index_dict=index_dict,
            max_output_tokens=max_output_tokens,
            max_index_memory=max_index_memory,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            name=name,
        )
        self.max_vector_memory = self.max_context
        self.prompt_func = self.vector_memory_prompt

    def vector_memory_prompt(
        self, message: str, k: int = 10
    ) -> Tuple[List[dict], dict]:
        """
        Combine system prompt, k most similar messages to the question, and the user prompt.

        :param question: A string representing the user question.
        :param k: The number of most similar messages to include in the prompt.
        :return: A tuple containing a list of strings as the prompt and the marked question.
        """
        sorted_messages, sorted_scores, sorted_indices = self.sorted_query(
            message, k=k, max_tokens=self.max_vector_memory, reverse=True
        )
        marked_question = mark_question(self.user_prompt(message))
        prompt = [mark_system(self.system_prompt)] + sorted_messages + [marked_question]
        return prompt, marked_question

    def weighted_memory_prompt(
        self,
        message: str,
        k: int = 10,
        decay_factor: float = 0.1,
        temporal_weight: float = 0.5,
    ) -> Tuple[List[dict], dict]:
        """
        Combine system prompt, weighted k most similar messages to the question, and the user prompt.

        :param question: A string representing the user question.
        :param k: The number of most similar messages to include in the prompt.
        :param decay_factor: A float representing the decay factor for weighting.
        :param temporal_weight: A float representing the weight of the temporal aspect.
        :return: A tuple containing a list of strings as the prompt and the marked question.
        """
        weighted_messages, weighted_scores, weighted_indices = self.weighted_query(
            message,
            k=k,
            max_tokens=self.max_vector_memory,
            decay_factor=decay_factor,
            temporal_weight=temporal_weight,
            order_by="chronological",
            reverse=True,
        )
        marked_question = mark_question(self.user_prompt(message))
        prompt = (
            [mark_system(self.system_prompt)] + weighted_messages + [marked_question]
        )
        return prompt, marked_question

    def query(self, question: str, verbose: bool = False, stream:bool = False) -> Union[Generator,str]:
        """
        Query the chatbot with a given question. The question is added to the memory, and the answer is returned
        and added to the memory.

        :param question: A string representing the user question.
        :param verbose: A boolean indicating whether to display input and output messages as Markdown.
        :return: A string representing the chatbot's response.
        """
                
        marked_question = mark_question(question)
        self.add_message(marked_question)
        answer = BaseChat.query(self, message=question, verbose=verbose, stream=stream)
        if stream:
            return answer
        else:
            self.add_message(answer)
            return answer


class FifoVectorChat(FifoThread, Chat):
    """
    A chatbot class that combines FIFO Memory Thread, Vector Memory Thread, BaseChat, and Prompter.
    The memory prompt is constructed by including both FIFO memory and Vector memory.
    """

    def __init__(
        self,
        model: str = None,
        index_dict: Optional[Dict[str, Union[PandasIndex, MemoryIndex]]] = None,
        system_prompt: str = None,
        user_prompt: str = None,
        name: str = "fifo_vector_memory",
        max_memory: int = 2048,
        max_index_memory: int = 400,
        max_output_tokens: int = 1000,
        longterm_thread: Optional[VectorThread] = None,
        longterm_frac: float = 0.5,
    ):
        self.total_max_memory = max_memory

        self.setup_longterm_memory(longterm_thread, max_memory, longterm_frac)
        FifoThread.__init__(
            self,
            name=name,
            max_memory=self.max_fifo_memory,
            longterm_thread=self.longterm_thread,
        )
        Chat.__init__(
            self,
            model=model,
            index_dict=index_dict,
            max_output_tokens=max_output_tokens,
            max_index_memory=max_index_memory,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            name=name,
        )
        
        self.prompt_func = self.fifovector_memory_prompt
        self.prompt_list = []

    def setup_longterm_memory(
        self,
        longterm_thread: Optional[VectorThread],
        max_memory: int,
        longterm_frac: float,
    ):
        """
        Set up long-term memory by allocating memory for the FIFO and Vector memory components.

        :param longterm_thread: An optional VectorThread for long-term memory.
        :param max_memory: The maximum amount of memory for the chatbot.
        :param longterm_frac: The fraction of memory dedicated to long-term memory.
        """
        if longterm_thread is None:
            self.longterm_frac = longterm_frac
            self.max_fifo_memory = int(max_memory * (1 - self.longterm_frac))
            self.max_vector_memory = max_memory - self.max_fifo_memory
            self.longterm_thread = VectorThread(
                name="longterm_memory", max_context=self.max_vector_memory
            )
        else:
            self.longterm_thread = longterm_thread
            self.max_vector_memory = self.longterm_thread.max_context
            self.max_fifo_memory = self.total_max_memory - self.max_vector_memory
            self.longterm_frac = self.max_vector_memory / self.total_max_memory

    def fifovector_memory_prompt(
        self, message: str, k: int = 10
    ) -> Tuple[List[dict], dict]:
        """
        Combine the system prompt, long-term memory (vector memory), short-term memory (FIFO memory), and the user prompt.

        :param question: A string representing the user question.
        :param k: The number of most similar messages to include from the long-term memory.
        :return: A tuple containing a list of strings as the prompt and the marked question.
        """
        prompt = [mark_system(self.system_prompt)]
        if (
            len(self.longterm_thread.memory_thread) > 0
            and self.longterm_thread.total_tokens <= self.max_vector_memory
        ):
            prompt += self.longterm_thread.memory_thread
        elif (
            len(self.longterm_thread.memory_thread) > 0
            and self.longterm_thread.total_tokens > self.max_vector_memory
        ):
            (
                sorted_messages,
                sorted_scores,
                sorted_indices,
            ) = self.longterm_thread.sorted_query(
                message, k=k, max_tokens=self.max_vector_memory, reverse=True
            )
            prompt += sorted_messages

        prompt += self.memory_thread
        marked_question = mark_question(self.user_prompt(message))
        prompt += [marked_question]
        return prompt, marked_question

    def query(self, question: str, verbose: bool = False, stream:bool = False) -> Union[Generator,str]:
        """
        Query the chatbot with a given question. The question is added to the memory, and the answer is returned
        and added to the memory.

        :param question: A string representing the user question.
        :param verbose: A boolean indicating whether to display input and output messages as Markdown.
        :return: A string representing the chatbot's response.
        """
        #marked_question = mark_question(question)
        original_question = {'role': 'user', 'content': question}
        prompt, marked_question = self.fifovector_memory_prompt(question)
        self.add_message(original_question)
        self.longterm_thread.add_message(original_question)
        self.add_message(marked_question)

        answer = BaseChat.query(self, message=question, verbose=verbose, stream=stream)
        if stream:
            return answer
        else:
            self.add_message(answer)
            self.longterm_thread.add_message(answer)
            return answer


class ContextManagedFifoVectorChat(FifoThread, Chat):
    def __init__(
        self,
        model: str = None,
        index_dict: Optional[Dict[str, Union[PandasIndex, MemoryIndex]]] = None,
        system_prompt: str = None,
        user_prompt: str = None,
        name: str = "fifo_vector_memory",
        max_memory: int = 2048,
        max_index_memory: int = 400,
        max_output_tokens: int = 1000,
        longterm_thread: Optional[VectorThread] = None,
        longterm_frac: float = 0.5,
    ):
        self.total_max_memory = max_memory

        self.setup_longterm_memory(longterm_thread, max_memory, longterm_frac)
        FifoThread.__init__(
            self,
            name=name,
            max_memory=self.max_fifo_memory,
            longterm_thread=self.longterm_thread,
        )
        Chat.__init__(
            self,
            model=model,
            index_dict=index_dict,
            max_output_tokens=max_output_tokens,
            max_index_memory=max_index_memory,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            name=name,
        )
        self.context_manager = ContextManager(index_dict)
        #self.prompt_func = self.fifovector_memory_prompt
        self.prompt_list = []
    
    def setup_longterm_memory(
        self,
        longterm_thread: Optional[VectorThread],
        max_memory: int,
        longterm_frac: float,
    ):
        """
        Set up long-term memory by allocating memory for the FIFO and Vector memory components.

        :param longterm_thread: An optional VectorThread for long-term memory.
        :param max_memory: The maximum amount of memory for the chatbot.
        :param longterm_frac: The fraction of memory dedicated to long-term memory.
        """
        #TODO preload longterm memory with index summaries
        if longterm_thread is None:
            self.longterm_frac = longterm_frac
            self.max_fifo_memory = int(max_memory * (1 - self.longterm_frac))
            self.max_vector_memory = max_memory - self.max_fifo_memory
            self.longterm_thread = VectorThread(
                name="longterm_memory", max_context=self.max_vector_memory
            )
        else:
            self.longterm_thread = longterm_thread
            self.max_vector_memory = self.longterm_thread.max_context
            self.max_fifo_memory = self.total_max_memory - self.max_vector_memory
            self.longterm_frac = self.max_vector_memory / self.total_max_memory

    def heat_trajectory(self, message, k=5):
        """
        This function gets the top k embeddings from all indexes including longterm, merges into numpy matrix,
        record boundaries, computes kernel adj matrix, and sums all connections in each index section.
        It returns the heat trajectory.
        """
        embeddings = []
        boundaries = [0]  # Start boundary
        top_k_hints = {}
        # Gather top k embeddings from each index
        logging.info("Computing Heat Trajectory for the current state of memory.")
        for index_key, index in self.index_dict.items():
            top_k, _, indices = index.token_bound_query(message, k=k, max_tokens=5000)
            top_k_embeddings = [index.embeddings[i] for i in indices]
            embeddings.extend(top_k_embeddings)
            boundaries.append(boundaries[-1] + k)
            top_k_hints[index_key] = top_k
        # Gather top k embeddings from longterm index
        logging.info(f"Number of values in Longterm Index: {len(self.longterm_thread.values)}")
        logging.info(f"Number of embeddings in Longterm Index: {len(self.longterm_thread.embeddings)}")

        if len(self.longterm_thread.values) >= k:
            if len(self.longterm_thread.embeddings) != len(self.longterm_thread.values):
                self.longterm_thread.compute_embeddings()
            top_k, _, indices = self.longterm_thread.token_bound_query(message, k=k, max_tokens=3000)
            top_k_embeddings = [self.longterm_thread.embeddings[i] for i in indices]
            embeddings.extend(top_k_embeddings)
            boundaries.append(boundaries[-1] + k)
            top_k_hints['longterm_thread'] =top_k
        # Convert list of embeddings into a numpy matrix
        embeddings_matrix = np.vstack(embeddings)
        adjacency_matrix = cosine_similarity(embeddings_matrix)
        # Add the message embedding to the top-k embeddings and normalize
        message_embedding = EMBEDDER.embed(data=message)  # Replace with your message embedding method
        top_k_embeddings_with_message = embeddings_matrix.copy()
        top_k_embeddings_with_message[:-1] += message_embedding
        norms = np.linalg.norm(top_k_embeddings_with_message, axis=1)
        normalized_embeddings = top_k_embeddings_with_message / norms[:, np.newaxis]
        
        # Compute the adjacency matrix using cosine similarity on the normalized embeddings
        adjacency_matrix_with_message = cosine_similarity(normalized_embeddings)
        #subtract the adjacency matrix from the adjacency matrix with message
        #adjacency_matrix_with_message = adjacency_matrix_with_message - adjacency_matrix
        adjacency_matrix_with_message = adjacency_matrix_with_message**2 - adjacency_matrix**2
        adjacency_matrix_with_message =  adjacency_matrix_with_message**2
        # Compute the stability of connections within each boundary
        boundary_stability = np.zeros(len(boundaries) - 1)
        for i in range(len(boundaries) - 1):
            boundary_connections = adjacency_matrix_with_message[boundaries[i]:boundaries[i+1], :][:, boundaries[i]:boundaries[i+1]]
            # apply a time decay param gamma to the boundary connections
            boundary_stability[i] = np.mean(boundary_connections)
            sigma = np.std(boundary_connections)
            boundary_stability[i] = boundary_stability[i] - sigma**2
        logging.info(f"Boundary Stability: {boundary_stability}")
        # Find the boundary with the highest stability

        # Compute heat trajectory by summing up all degrees in each index section
        degrees = np.sum(adjacency_matrix_with_message, axis=1)
        #logging.info(f"Degrees: {degrees}")
        heat_trajectory = [np.sum(degrees[boundaries[i]:boundaries[i + 1]]) for i in range(len(boundaries) - 1)]
        logging.info(f"Heat Trajectory: {heat_trajectory}")
        # Return dictionary of connections mapped to index names and the max boundary index

        heat_dict = dict(zip(list(self.index_dict.keys()) + ['longterm_thread'], heat_trajectory))
        # values should sum to 1
        sum_of_vals = sum(heat_dict.values())
        heat_dict = {k: v / sum_of_vals for k, v in heat_dict.items()}
        return heat_dict, top_k_hints


    def fifovector_memory_prompt(
        self, message: str, k: int = 3
    ) -> Tuple[List[dict], dict]:
        #index_name = self.context_manager.get_context_for_user_input(message)
        hdict, top_k_hint_dict = self.heat_trajectory(message)
        logging.info(f"Heat Dictionary: {hdict}")
        #get index with max heat
        #sort keys by value the min val should be the 0 index
        hdict = {k: v for k, v in sorted(hdict.items(), key=lambda item: item[1])}
        min_heat_index = list(hdict.keys())[0]
        second_min_heat_index = list(hdict.keys())[1]

        logging.info(f"Max Heat Index: {min_heat_index}")
        if min_heat_index == 'longterm_thread':
            logging.info(f"Chosen Index: {min_heat_index} - Retrieving prompt from long-term memory.")
            logging.info(f"Number of values in Index: {len(self.longterm_thread.values)}")
            top_k_hint = top_k_hint_dict[min_heat_index][:1] + top_k_hint_dict[second_min_heat_index][:k-1]
            logging.info(f"Top K Hint: {top_k_hint}")
            prompt =f'[LONG TERM MEMORY]{str(top_k_hint)}\n\n [QUESTION]: {message}'
        elif min_heat_index in self.index_dict.keys():
            # if the difference between min and second min is less than 0.1, merge hints from both indexes
            if hdict[min_heat_index] - hdict[second_min_heat_index] < 0.1:
                logging.info(f"Chosen Index: {min_heat_index} - Retrieving prompt from index.")
                logging.info(f"Number of values in Index {self.index_dict[min_heat_index].name}: {len(self.index_dict[min_heat_index].values)}")
                #take 2 from first index topk and 1 from second index topk
                top_k_hint = top_k_hint_dict[min_heat_index][:k-1] + top_k_hint_dict[second_min_heat_index][:1]
                logging.info(f"Top K Hint: {top_k_hint}")
                prompt =f'{str(top_k_hint)}\n\n [QUESTION]: {message}'
            else:
                logging.info(f"Chosen Index: {min_heat_index} - Retrieving prompt from index.")
                logging.info(f"Number of values in Index {self.index_dict[min_heat_index].name}: {len(self.index_dict[min_heat_index].values)}")
                top_k_hint = top_k_hint_dict[min_heat_index][:k]
                logging.info(f"Top K Hint: {top_k_hint}")
                prompt =f'{str(top_k_hint)}\n\n [QUESTION]: {message}'
        else:
            raise ValueError("The provided index name is not available.")

        return prompt

    def context_query(self, question: str, verbose: bool = False, stream: bool = False) -> Union[Generator, str]:
        prompt = self.fifovector_memory_prompt(question)

        original_question = {'role': 'user', 'content': question}
        modified_question = mark_question(prompt)
        self.add_message(original_question)
        self.add_message(modified_question)
        #self.longterm_thread.add_message(modified_question)

        answer = BaseChat.query(self, message=prompt, verbose=verbose, stream=stream)

        if stream:
            return answer
        else:
            self.add_message(answer)
            #self.longterm_thread.add_message(answer))
            return answer
