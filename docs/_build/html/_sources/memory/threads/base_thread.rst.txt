base_thread
===========

.. code-block:: python

	
	
	class BaseThread:
	    """
	    This class is used to keep track of the memory thread of a conversation and the total number of tokens.
	    All conversation memories should subclass this class. If max_memory is None, it has
	    no limit to the number of tokens that can be stored in the memory thread.
	    """
	
	    def __init__(
	        self,
	        name: str = "memory",
	        max_memory: Optional[int] = None,
	        tokenizer: Optional[Any] = None,
	    ) -> None:
	        """
	        Initialize the BaseThread instance.
	
	        :param name: The name of the memory thread. Defaults to 'memory'.
	        :param max_memory: The maximum number of tokens allowed in the memory thread.
	                           Defaults to None, which means no limit.
	        :param tokenizer: The tokenizer to be used for tokenizing messages.
	                          Defaults to None, which means using the tiktoken encoding for the 'gpt-3.5-turbo' model.
	        """
	        self.name = name
	        self.max_memory = max_memory
	        self.memory_thread = []
	        self.time_stamps = []
	        self.message_tokens = []
	        self.total_tokens = 0
	        if tokenizer is None:
	            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
	
	    def __getitem__(self, idx):
	        return self.memory_thread[idx]
	
	    def __len__(self):
	        return len(self.memory_thread)
	
	    def reset_memory(self) -> None:
	        """
	        Reset the memory thread.
	        """
	        self.memory_thread = []
	        self.time_stamps = []
	        self.message_tokens = []
	        self.total_tokens = 0
	
	    def get_message_tokens(self, message_dict: dict) -> int:
	        """
	        Calculate the number of tokens in a message, including the role token.
	
	        :param message_dict: A dictionary containing the role and content of the message.
	        :return: The total number of tokens in the message.
	        """
	        message_dict = check_dict(message_dict)
	        message = message_dict["content"]
	        return len(self.tokenizer.encode(message)) + 6  # +6 for the role token
	
	    def get_message_role(self, message_dict: dict) -> str:
	        """
	        Get the role of the message from a message dictionary.
	
	        :param message_dict: A dictionary containing the role and content of the message.
	        :return: The role of the message.
	        """
	        message_dict = check_dict(message_dict)
	        return message_dict["role"]
	
	    def add_message(self, message_dict: dict) -> None:
	        """
	        Add a message to the memory thread.
	
	        :param message_dict: A dictionary containing the role and content of the message.
	        """
	        message_tokens = self.get_message_tokens(message_dict)
	
	        if (
	            self.max_memory is None
	            or self.total_tokens + message_tokens <= self.max_memory
	        ):
	            # add the message_dict to the memory_thread
	            # update the total number of tokens
	            self.memory_thread.append(message_dict)
	            self.total_tokens += message_tokens
	            self.message_tokens.append(message_tokens)
	            time_stamp = time.time()
	            self.time_stamps.append(time_stamp)
	        else:
	            display(
	                Markdown(
	                    "The memory BaseThread is full, the last message was not added"
	                )
	            )
	
	    def remove_message(
	        self, message_dict: Union[dict, None] = None, idx: Union[int, None] = None
	    ) -> None:
	        """
	        Remove a message from the memory thread.
	        """
	        if message_dict is None and idx is None:
	            raise Exception("You need to provide either a message_dict or an idx")
	        elif message_dict is not None and idx is not None:
	            raise Exception("You need to provide either a message_dict or an idx")
	
	        if idx is None:
	            message_dict = check_dict(message_dict)
	            search_results = self.find_message(message_dict)
	            if search_results is not None:
	                idx = search_results[-1]["idx"]
	                message = search_results[-1]["message_dict"]
	                self.memory_thread.pop(idx)
	                self.message_tokens.pop(idx)
	                self.time_stamps.pop(idx)
	                self.total_tokens -= self.get_message_tokens(message)
	            else:
	                raise Exception("The message was not found in the memory BaseThread")
	        else:
	            if idx < len(self.memory_thread):
	                message = self.memory_thread.pop(idx)
	                self.total_tokens -= self.get_message_tokens(message)
	            else:
	                raise Exception("The index was out bound")
	
	    def find_message(
	        self, message: Union[dict, str], role: Union[str, None] = None
	    ) -> Union[None, list]:
	        """
	        Find a message in the memory thread. If the message is a dictionary, it will search for the exact match.
	        If the message is a string, it will search for the string in the content of the message dictionary."""
	        # check if the message is a dictioanry or a string
	        message = message if isinstance(message, str) else check_dict(message)
	        search_results = []
	        for idx, message_dict in enumerate(self.memory_thread):
	            target = (
	                message_dict if isinstance(message, dict) else message_dict["content"]
	            )
	            if target == message and (role is None or message_dict["role"] == role):
	                search_results.append({"idx": idx, "message_dict": message_dict})
	        return search_results if len(search_results) > 0 else None
	
	    def find_role(self, role: str) -> Union[None, list]:
	        """
	        Find all messages with a specific role in the memory thread.
	        """
	        search_results = []
	        for idx, message_dict in enumerate(self.memory_thread):
	            if message_dict["role"] == role:
	                search_results.append({"idx": idx, "message_dict": message_dict})
	        return search_results if len(search_results) > 0 else None
	
	    def last_message(self, role: Union[str, None] = None) -> Union[None, dict]:
	        """
	        Get the last message in the memory thread with a specific role."""
	        if role is None:
	            return self.memory_thread[-1]
	        else:
	            for message_dict in reversed(self.memory_thread):
	                if message_dict["role"] == role:
	                    return message_dict
	            return None
	
	    def first_message(self, role: Union[str, None] = None) -> Union[None, dict]:
	        """
	        Get the first message in the memory thread with a specific role."""
	        if role is None:
	            return self.memory_thread[0]
	        else:
	            for message_dict in self.memory_thread:
	                if message_dict["role"] == role:
	                    return message_dict
	            return None
	
	    def messages_before(
	        self, message: dict, role: Union[str, None] = None
	    ) -> Union[None, list]:
	        """
	        Get all messages before a specific message in the memory thread with a specific role."""
	        messages = []
	        # print("ci siamo")
	        for idx, message_dict in enumerate(self.memory_thread):
	            # print(message, message_dict)
	            if message_dict == message :
	                for mess in self.memory_thread[:idx]:
	                    if role is None or mess["role"] == role:
	                        messages.append(mess)
	                break
	        return messages if len(messages) > 0 else None
	    
	    def messages_after(
	        self, message: dict, role: Union[str, None] = None
	    ) -> Union[None, list]:
	        """
	        Get all messages after a specific message in the memory thread with a specific role."""
	        messages = []
	        for idx, message_dict in enumerate(self.memory_thread):
	            if message_dict == message:
	                for mess in self.memory_thread[:idx]:
	                    if role is None or mess["role"] == role:
	                        messages.append(mess)
	                break
	        return messages if len(messages) > 0 else None
	
	    def messages_between(
	        self, start_message: dict, end_message: dict, role: Union[str, None] = None
	    ) -> Union[None, list]:
	        """
	        Get all messages between two specific messages in the memory thread with a specific role."""
	        messages = []
	        for idx, message_dict in enumerate(self.memory_thread):
	            if message_dict == start_message:
	                start_idx = idx
	                break
	        for idx, message_dict in enumerate(self.memory_thread):
	            if message_dict == end_message:
	                end_idx = idx
	                break
	        for mess in self.memory_thread[start_idx + 1 : end_idx-1]:
	                    if role is None or mess["role"] == role:
	                        messages.append(mess)
	        return messages if len(messages) > 0 else None
	
	    def messages_more_tokens(self, tokens: int, role: Union[str, None] = None):
	        """
	        Get all messages with more tokens than a specific number in the memory thread with a specific role."""
	        messages = []
	        for idx, message_dict in enumerate(self.memory_thread):
	            if self.message_tokens[idx] > tokens and (
	                role is None or message_dict["role"] == role
	            ):
	                messages.append(message_dict)
	        return messages if len(messages) > 0 else None
	
	    def messages_less_tokens(self, tokens: int, role: Union[str, None] = None):
	        """
	        Get all messages with less tokens than a specific number in the memory thread with a specific role."""
	        messages = []
	        for idx, message_dict in enumerate(self.memory_thread):
	            if self.message_tokens[idx] < tokens and (
	                role is None or message_dict["role"] == role
	            ):
	                messages.append(message_dict)
	        return messages if len(messages) > 0 else None
	
	    def messages_between_tokens(
	        self, start_tokens: int, end_tokens: int, role: Union[str, None] = None
	    ):
	        """
	        Get all messages with less tokens than a specific number in the memory thread with a specific role."""
	        messages = []
	        for idx, message_dict in enumerate(self.memory_thread):
	            if (
	                self.message_tokens[idx] > start_tokens
	                and self.message_tokens[idx] < end_tokens
	                and (role is None or message_dict["role"] == role)
	            ):
	                messages.append(message_dict)
	        return messages if len(messages) > 0 else None
	
	    def messages_before_time(self, time_stamp, role: Union[str, None] = None):
	        """
	        Get all messages before a specific time in the memory thread with a specific role."""
	        messages = []
	        for idx, message_dict in enumerate(self.memory_thread):
	            if self.time_stamps[idx] < time_stamp and (
	                role is None or message_dict["role"] == role
	            ):
	                messages.append(message_dict)
	        return messages if len(messages) > 0 else None
	
	    def messages_after_time(self, time_stamp, role: Union[str, None] = None):
	        """
	        Get all messages after a specific time in the memory thread with a specific role."""
	        messages = []
	        for idx, message_dict in enumerate(self.memory_thread):
	            if self.time_stamps[idx] > time_stamp and (
	                role is None or message_dict["role"] == role
	            ):
	                messages.append(message_dict)
	        return messages if len(messages) > 0 else None
	
	    def messages_between_time(
	        self, start_time, end_time, role: Union[str, None] = None
	    ):
	        """
	        Get all messages between two specific times in the memory thread with a specific role."""
	        messages = []
	        for idx, message_dict in enumerate(self.memory_thread):
	            if (
	                self.time_stamps[idx] > start_time
	                and self.time_stamps[idx] < end_time
	                and (role is None or message_dict["role"] == role)
	            ):
	                messages.append(message_dict)
	        return messages if len(messages) > 0 else None
	
	    def token_bound_history(
	        self, max_tokens: int, max_history=None, role: Union[str, None] = None
	    ):
	        messages = []
	        indices = []
	        tokens = 0
	        if max_history is None:
	            max_history = len(self.memory_thread)
	
	        for idx, message_dict in enumerate(reversed(self.memory_thread)):
	            if  tokens + self.message_tokens[idx] <= max_tokens:
	                if role is not None and message_dict["role"] != role:
	                    continue
	                messages.append(message_dict)
	                indices.append(len(self.memory_thread) - 1 - idx)
	                tokens += self.message_tokens[idx]
	            else:
	                break
	        return messages, indices if len(messages) > 0 else (None, None)
	

.. automodule:: base_thread
   :members:
