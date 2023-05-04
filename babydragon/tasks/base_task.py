import copy
import json
import os
import time
from typing import Any, List

from babydragon.utils.multithreading import (RateLimitedThreadPoolExecutor,
                                             RateLimiter)


class BaseTask:
    def __init__(
        self,
        path: List[List[int]],
        max_workers: int = 1,
        task_id: str = "task",
        calls_per_minute: int = 20,
        backup: bool = True,
    ):
        self.task_id = task_id
        self.path = path
        self.results = [None] * len(self.path)
        self.max_workers = max_workers
        self.parallel = True if max_workers > 1 else False
        self.rate_limiter = RateLimiter(calls_per_minute)
        self.failed_sub_tasks = []
        self.backup = backup

    def _save_results_to_file(self) -> None:
        with open(f"{self.task_id}_results.json", "w") as f:
            json.dump(self.results, f)

    def _load_results_from_file(self) -> None:
        if os.path.exists(f"{self.task_id}_results.json"):
            try:
                with open(f"{self.task_id}_results.json", "r") as f:
                    self.results = json.load(f)
                    print(f"Loaded {len(self.results)} results from file.")
            except Exception as e:
                print(f"Error loading results from file: {e}")
                print("Starting from scratch.")
        else:
            print("No results file found, starting from scratch.")

    def _execute_sub_task(self, sub_path: List[int]) -> List[str]:
        sub_results = []
        for i in sub_path:
            response = "Implement the response function in the subclass"
            sub_results.append(response)
        return sub_results

    def execute_task(self) -> None:
        if self.backup:
            self._load_results_from_file()

        with RateLimitedThreadPoolExecutor(
            max_workers=self.max_workers,
            calls_per_minute=self.rate_limiter.calls_per_minute,
        ) as executor:
            futures = []
            print(f"Executing task {self.task_id} using {self.max_workers} workers.")

            for i, sub_path in enumerate(self.path):
                if self.results[i] is not None:
                    pass
                else:
                    future = executor.submit(self._execute_sub_task, sub_path)
                    futures.append((i, future))

            for i, future in futures:
                try:
                    execution_start_time = time.time()
                    sub_task_result = future.result()
                    execution_end_time = time.time()
                    print(
                        f"Sub-task {i} executed in {execution_end_time - execution_start_time:.2f} seconds."
                    )

                    save_start_time = time.time()
                    self.results[i] = sub_task_result
                    # self.results.append(sub_task_result)
                    if self.backup:
                        self._save_results_to_file()
                    save_end_time = time.time()
                    print(
                        f"Sub-task {i} results saved in {save_end_time - save_start_time:.2f} seconds."
                    )
                except Exception as e:
                    print(f"Error in sub-task {i}: {e}")
                    default_result = f"Error in sub-task {i}: {e}"
                    self.results[i] = default_result
                    self._save_results_to_file()
                    self.failed_sub_tasks.append((self.path[i], str(e)))

                except KeyboardInterrupt:
                    print("Keyboard interrupt detected, stopping task execution.")
                    executor.shutdown(wait=False)
                    break

        print("Task execution completed.")

    def work(self) -> List[Any]:
        self.execute_task()
        work = []
        for sub_result in self.results:
            for index_id, response in sub_result.items():
                work.append((index_id, response))
        # sort the content to write by index_id
        work.sort(key=lambda x: int(x[0]))
        return work
