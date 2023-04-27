
import copy
from typing import List, Any
from babydragon.chat.chat import Chat
from babydragon.memory.indexes.memory_index import MemoryIndex
from babydragon.memory.threads.base_thread import BaseThread
from babydragon.tasks.base_task import BaseTask

class LLMReader(BaseTask):
    def __init__(self, index: MemoryIndex, path: List[List[int]], chatbot: Chat,read_func: None, max_workers: int = 4):
        """
        Initialize a LLMReadTask instance.

        :param index: List of strings representing the queries.
        :param path: List of lists, each sub-list defines a sequence over which the task is executed.
        :param chatbot: Chatbot instance used for executing queries.
        :param max_workers: Maximum number of worker threads (default is 4).
        """
        BaseTask.__init__(self,index, path, max_workers)
        self.chatbot = chatbot
        self.read_func = read_func if read_func else self.llm_response

    @staticmethod
    def llm_response(chatbot: Chat,message: str, string_out = False):
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
            #copy the chatbot instance and resets the memory before making the queries in case of multi-threading
            chatbot_instance = copy.deepcopy(self.chatbot)
        else:
            chatbot_instance = self.chatbot
        if isinstance(self.chatbot, BaseThread):
            chatbot_instance.reset_memory()
        
        sub_results = []
        for i in sub_path:
            response = self.read_func(chatbot_instance,self.index[i])
            sub_results.append(response)
        return sub_results
    
    def read(self):
        self.execute_task()
        return self.results
    
    def reset_chat_memory(self) -> None:
        """
        Reset the chatbot's memory.
        """
        #TODO

class LLMWriter(BaseTask):
    def __init__(self, index: MemoryIndex, path: List[List[int]], chatbot: Chat, write_func: None, max_workers: int = 4):
        """
        Initialize a LLMWriteTask instance.

        :param index: List of strings representing the queries.
        :param path: List of lists, each sub-list defines a sequence over which the task is executed.
        :param chatbot: Chatbot instance used for executing queries.
        :param max_workers: Maximum number of worker threads (default is 4).
        """
        BaseTask.__init__(self,index, path, max_workers)
        self.chatbot = chatbot
        self.write_func = write_func if write_func else self.llm_response
        self.new_index = copy.deepcopy(self.index)
        self.new_index.name = self.index.name + "_new"

    @staticmethod
    def llm_response(chatbot: Chat,message: str):
        return chatbot.reply(message), True

    def _execute_sub_task(self, sub_path: List[int]) -> List[str]:
        """
        Execute a sub-task using a separate copy of the chatbot instance.

        :param sub_path: List of indices representing the sub-task's sequence.
        :return: List of strings representing the responses for each query in the sub-task.
        """
        if self.parallel:
            #copy the chatbot instance and resets the memory before making the queries in case of multi-threading
            chatbot_instance = copy.deepcopy(self.chatbot)
        else:
            chatbot_instance = self.chatbot
        if isinstance(self.chatbot, BaseThread):
            chatbot_instance.reset_memory()
        
        sub_results = {}
        for i in sub_path:
            response, write = self.write_func(chatbot_instance,self.index[i])
            if write:
                sub_results[i] = response 
        return sub_results
    
    def write(self):
        self.execute_task()
        # for each subresults calls self.index.subsitute_at_index(index, sub_result at index)
        for sub_result in self.results:
            for index, response in sub_result.items():
                self.new_index.substitute_at_index(index, response)
        self.new_index.save()
        return self.new_index    
                




