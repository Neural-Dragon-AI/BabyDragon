import copy
from typing import Any, List
from babydragon.chat.chat import Chat
from babydragon.memory.indexes.memory_index import MemoryIndex
from babydragon.memory.threads.base_thread import BaseThread
from babydragon.tasks.base_task import BaseTask


class LLMReader(BaseTask):
    def __init__(
        self,
        index: MemoryIndex,
        path: List[List[int]],
        chatbot: Chat,
        read_func = None,
        max_workers: int = 1,
        task_id: str = "LLMReadTask",
        calls_per_minute: int = 20,
    ):
        """
        Initialize a LLMReadTask instance.

        :param index: List of strings representing the queries.
        :param path: List of lists, each sub-list defines a sequence over which the task is executed.
        :param chatbot: Chatbot instance used for executing queries.
        :param max_workers: Maximum number of worker threads (default is 4).
        """
        BaseTask.__init__(self, index, path, max_workers, task_id, calls_per_minute)
        self.chatbot = chatbot
        self.read_func = read_func if read_func else self.llm_response

    def llm_response(chatbot: Chat, message: str, string_out=False):
        if string_out:
            return chatbot.reply(message)
        return chatbot.query(message)

    def _execute_sub_task(self, sub_path: List[int]) -> List[str]:
        """
        Execute a sub-task using a separate copy of the chatbot instance. each sub-stasks uses a
        a clean memory instance.

        :param sub_path: List of indices representing the sub-task's sequence.
        :return: List of strings representing the responses for each query in the sub-task.
        """
        if self.parallel:
            # copy the chatbot instance and resets the memory before making the queries in case of multi-threading
            chatbot_instance = copy.deepcopy(self.chatbot)
        else:
            chatbot_instance = self.chatbot
        if isinstance(self.chatbot, BaseThread):
            chatbot_instance.reset_memory()

        sub_results = []
        for i in sub_path:
            response = self.read_func(chatbot_instance, self.index.values[i])
            sub_results.append(response)
        return sub_results

    def read(self):
        self.execute_task()
        return self.results


class LLMWriter(BaseTask):
    def __init__(
        self,
        index: MemoryIndex,
        path: List[List[int]],
        chatbot: Chat,
        write_func = None,
        context= None,
        task_name="summary",
        max_workers: int = 1,
        task_id: str = "LLMWriteTask",
        calls_per_minute: int = 20,
    ):
        """
        Initialize a LLMWriteTask instance.

        :param index: List of strings representing the queries.
        :param path: List of lists, each sub-list defines a sequence over which the task is executed.
        :param chatbot: Chatbot instance used for executing queries.
        :param max_workers: Maximum number of worker threads (default is 4).
        """
        BaseTask.__init__(self, index, path, max_workers, task_id, calls_per_minute)
        self.chatbot = chatbot
        self.write_func = write_func if write_func else self.llm_response
        self.new_index_name = self.index.name + f"_{task_name}"
        self.context = context

    @staticmethod
    def llm_response(chatbot: Chat, message: str, context = None, id = None):
        return chatbot.reply(message)

    def _execute_sub_task(self, sub_path: List[int]) -> List[str]:
        """
        Execute a sub-task using a separate copy of the chatbot instance.

        :param sub_path: List of indices representing the sub-task's sequence.
        :return: List of strings representing the responses for each query in the sub-task.
        """
        if self.parallel:
            # copy the chatbot instance and resets the memory before making the queries in case of multi-threading
            chatbot_instance = copy.deepcopy(self.chatbot)
        else:
            chatbot_instance = self.chatbot
        if isinstance(self.chatbot, BaseThread):
            chatbot_instance.reset_memory()

        sub_results = {}
        for i in sub_path:
            current_val = self.index.values[i]
            response = self.write_func(chatbot_instance, current_val, self.context, id = i)
            sub_results[i] = response
        return sub_results

    def write(self):
        self.execute_task()
        content_to_write = []
        for sub_result in self.results:
            for index_id, response in sub_result.items():
                content_to_write.append((index_id, response))
        # sort the content to write by index_id
        content_to_write.sort(key=lambda x: x[0])
        self.new_index = MemoryIndex(name=self.new_index_name)
        self.new_index.init_index(values=[x[1] for x in content_to_write])
        self.new_index.save()
        return self.new_index
