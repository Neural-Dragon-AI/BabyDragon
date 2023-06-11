import re
from typing import Dict, List, Optional, Tuple, Union
from babydragon.memory.indexes.memory_index import MemoryIndex

class ContextManager:
    """
    A class that manages the context of user interactions and maps them to the appropriate memory index or thread.
    """
    def __init__(
        self, 
        index_dict: Dict[str, MemoryIndex], 
        longterm_thread_keywords: List[str] = ["long ago", "in the past", "long term", "long-term", "longterm"]
    ) -> None:
        """
        Initialize the ContextManager with the available indexes.

        :param index_dict: A dictionary mapping index names to MemoryIndex instances.
        :param fifo_thread_keywords: A list of keywords indicating user reference to recent conversation context.
        :param longterm_thread_keywords: A list of keywords indicating user reference to long-term conversation context.
        """
        self.index_dict = index_dict
        self.longterm_thread_keywords = longterm_thread_keywords
        self.keyword_to_index_map = self.create_keyword_to_index_map()

    def create_keyword_to_index_map(self) -> Dict[str, str]:
        """
        Create a mapping from keywords to index names.

        :return: A dictionary mapping keywords to index names or threads.
        """
        keyword_to_index_map = {index_name.split('_')[0]: index_name for index_name in self.index_dict.keys()}
        for keyword in self.longterm_thread_keywords:
            keyword_to_index_map[keyword] = "longterm_thread"
        return keyword_to_index_map

    def get_context_for_user_input(self, user_input: str) -> str:
        """
        Get the appropriate context (index or thread) for the given user input.

        :param user_input: A string representing the user's input.
        :return: The name of the appropriate context (index or thread), or None if no specific context is appropriate.
        """
        user_input_str = str(user_input)  # Ensure user_input is a string
        for keyword, context in self.keyword_to_index_map.items():
            if re.search(r'\b' + keyword + r'\b', user_input_str, re.I):
                return context
        return None

