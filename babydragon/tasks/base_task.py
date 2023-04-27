import copy
from typing import List, Any
from concurrent.futures import ThreadPoolExecutor
from babydragon.chat.chat import Chat

class BaseTask:
    def __init__(self, index: List[str], path: List[List[int]], chatbot: Chat, max_workers: int = 4):
        """
        Initialize a BaseTask instance.

        :param index: List of strings representing the queries.
        :param path: List of lists, each sub-list defines a sequence over which the task is executed.
        :param chatbot: Chatbot instance used for executing queries.
        :param max_workers: Maximum number of worker threads (default is 4).
        """
        self.index = index
        self.path = path
        self.chatbot = chatbot
        self.results = []
        self.max_workers = max_workers

    def _execute_sub_task(self, sub_path: List[int]) -> List[str]:
        """
        Execute a sub-task using a separate copy of the chatbot instance.

        :param sub_path: List of indices representing the sub-task's sequence.
        :return: List of strings representing the responses for each query in the sub-task.
        """
        if hasattr(self.chatbot, "memory_thread"):
            chatbot_instance = copy.deepcopy(self.chatbot)
        else:
            self.reset_chat_memory()
            chatbot_instance = self.chatbot
        sub_results = []
        for i in sub_path:
            response = chatbot_instance.query(self.index[i])
            sub_results.append(response)
        return sub_results

    def execute_task(self, parallel = False) -> None:
        """
        Execute the task by concurrently processing sub-tasks using worker threads.
        """
        if parallel:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                self.results = list(executor.map(self._execute_sub_task, self.path))
        else:
            for sub_path in self.path:
                self.results.append(self._execute_sub_task(sub_path))

    def reset_chat_memory(self) -> None:
        """
        Reset the chatbot's memory.
        """
        #TODO
