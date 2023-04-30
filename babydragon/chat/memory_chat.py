from typing import Dict, List, Optional, Tuple, Union

from babydragon.chat.chat import BaseChat, Chat, Prompter
from babydragon.memory.indexes.memory_index import MemoryIndex
from babydragon.memory.indexes.pandas_index import PandasIndex
from babydragon.memory.threads.base_thread import BaseThread
from babydragon.memory.threads.fifo_thread import FifoThread
from babydragon.memory.threads.vector_thread import VectorThread
from babydragon.utils.oai import mark_answer, mark_question, mark_system


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

    def query(self, question: str, verbose: bool = True) -> str:
        """
        Query the chatbot with a given question. The question is added to the memory, and the answer is returned
        and added to the memory.

        :param question: A string representing the user question.
        :param verbose: A boolean indicating whether to display input and output messages as Markdown.
        :return: A string representing the chatbot's response.
        """
        # First call the base class's query method
        answer = BaseChat.query(self, message=question, verbose=verbose)
        marked_question = mark_question(question)
        # Add the marked question and answer to the memory
        self.add_message(marked_question)
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

    def query(self, question: str, verbose: bool = False) -> str:
        """
        Query the chatbot with a given question. The question is added to the memory, and the answer is returned
        and added to the memory.

        :param question: A string representing the user question.
        :param verbose: A boolean indicating whether to display input and output messages as Markdown.
        :return: A string representing the chatbot's response.
        """
        # First call the base class's query method
        answer = BaseChat.query(self, message=question, verbose=verbose)
        marked_question = mark_question(question)
        # Add the marked question and answer to the memory
        self.add_message(marked_question)
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

    def query(self, question: str, verbose: bool = False) -> str:
        """
        Query the chatbot with a given question. The question is added to the memory, and the answer is returned
        and added to the memory.

        :param question: A string representing the user question.
        :param verbose: A boolean indicating whether to display input and output messages as Markdown.
        :return: A string representing the chatbot's response.
        """
        answer = BaseChat.query(self, message=question, verbose=verbose)
        marked_question = mark_question(question)
        self.add_message(marked_question)
        self.add_message(answer)
        return answer
