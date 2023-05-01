from typing import Dict, Optional, Union

from babydragon.chat.base_chat import BaseChat, Prompter
from babydragon.chat.prompts.default_prompts import (INDEX_HINT_PROMPT,
                                                     INDEX_SYSTEM_PROMPT,
                                                     QUESTION_INTRO)
from babydragon.memory.indexes.memory_index import MemoryIndex


class Chat(BaseChat, Prompter):
    """
    This class combines the BaseChat and Prompter classes to create a oneshot chatbot with a system and user prompt,
    and the ability to handle multiple index_dict.
    """

    def __init__(
        self,
        model: str = None,
        max_output_tokens: int = 1000,
        system_prompt: str = None,
        user_prompt: str = None,
        index_dict: Optional[Dict[str,MemoryIndex]] = None,
        max_index_memory: int = 1000,
    ) -> None:
        BaseChat.__init__(self, model=model, max_output_tokens=max_output_tokens)
        Prompter.__init__(self, system_prompt=system_prompt, user_prompt=user_prompt)
        self.index_dict = index_dict
        self.setup_indices(max_index_memory)

    def setup_indices(self, max_index_memory):
        """setup the index_dict for the chatbot. Change the system and user prompts to the index prompts if they are not user defined if there is an index."""
        if self.index_dict is not None:
            self.current_index = list(self.index_dict.keys())[0]
            self.system_prompt = (
                INDEX_SYSTEM_PROMPT
                if self.user_defined_system_prompt is False
                else self.system_prompt
            )
            self.user_prompt = (
                self.get_index_hints
                if self.user_defined_user_prompt is False
                else self.user_prompt
            )
            self.max_index_memory = max_index_memory

    def get_index_hints(
        self, question: str, k: int = 10, max_tokens: int = None
    ) -> str:
        """
        Get hints from the current index for the given question.

        :param question: A string representing the user question.
        :param k: The number of most similar messages to include from the index.
        :param max_tokens: The maximum number of tokens to be retrieved from the index.
        :return: A string representing the hint prompt with the question.
        """
        if max_tokens is None:
            max_tokens = self.max_index_memory
        hints = []
        if self.current_index is not None:
            index_instance = self.index_dict[self.current_index]
            if isinstance(index_instance, MemoryIndex):
                hints, _, _ = index_instance.token_bound_query(
                    question, k=k, max_tokens=max_tokens
                )
            else:
                raise ValueError("The current index is not a valid index instance.")
            hints_string = "\n".join(hints)
            hint_prompt = INDEX_HINT_PROMPT
            question_intro = QUESTION_INTRO
            return hint_prompt.format(
                hints_string=hints_string
            ) + question_intro.format(question=question)
        else:
            return question

    def set_current_index(self, index_name: Optional[str]) -> None:
        """
        Set the current index to be used for hints.

        :param index_name: A string representing the index name or None to clear the current index.
        :raise ValueError: If the provided index name is not available.
        """
        if self.index_dict is None:
            raise ValueError("No index_dict are available.")
        elif index_name in self.index_dict:
            self.current_index = index_name
        elif index_name is None:
            self.current_index = None
        else:
            raise ValueError("The provided index name is not available.")
