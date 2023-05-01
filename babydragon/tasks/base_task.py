import copy
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List

from babydragon.memory.indexes.memory_index import MemoryIndex


class BaseTask:
    def __init__(self, index: MemoryIndex, path: List[List[int]], max_workers: int = 1):
        """
        Initialize a BaseTask instance.

        :param index: List of strings representing the queries.
        :param path: List of lists, each sub-list defines a sequence over which the task is executed.
        :param max_workers: Maximum number of worker threads (default is 4).
        """
        self.index = index
        self.path = path
        self.results = []
        self.max_workers = max_workers
        self.parallel = True if max_workers > 1 else False

    def _execute_sub_task(self, sub_path: List[int]) -> List[str]:
        """
        to be implemented by subclasses:

        :param sub_path: List of indices representing the sub-task's sequence.
        :return: List of strings representing the responses for each query in the sub-task.
        """

        sub_results = []
        for i in sub_path:
            response = self.index[i]
            sub_results.append(response)
        return sub_results

    def execute_task(self) -> None:
        """
        Execute the task by concurrently processing sub-tasks using worker threads.
        """
        
        for sub_path in self.path:
            self.results.append(self._execute_sub_task(sub_path))
