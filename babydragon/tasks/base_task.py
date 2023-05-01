import copy
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List
import json
from babydragon.memory.indexes.memory_index import MemoryIndex


class BaseTask:
    def __init__(self, index: MemoryIndex, path: List[List[int]], max_workers: int = 1, task_id: str = "task"):
        self.task_id = task_id
        self.index = index
        self.path = path
        self.results = []
        self.max_workers = max_workers
        self.parallel = True if max_workers > 1 else False

    def _save_results_to_file(self) -> None:
        with open(f"{self.task_id}_results.json", "w") as f:
            json.dump(self.results, f)

    def _load_results_from_file(self) -> None:
        if os.path.exists(f"{self.task_id}_results.json"):
            with open(f"{self.task_id}_results.json", "r") as f:
                self.results = json.load(f)
                print(f"Loaded {len(self.results)} results from file.")
        else:
            print("No results file found, starting from scratch.")

    def _execute_sub_task(self, sub_path: List[int]) -> List[str]:
        sub_results = []
        for i in sub_path:
            response = self.index[i]
            sub_results.append(response)
        return sub_results

    def execute_task(self) -> None:
        self._load_results_from_file()

        for i, sub_path in enumerate(self.path):
            if i < len(self.results):
                print(f"Sub-task {i} already completed, skipping...")
            else:
                print(f"Executing sub-task {i}...")
                sub_task_result = self._execute_sub_task(sub_path)
                self.results.append(sub_task_result)
                self._save_results_to_file()

        print("Task execution completed.")