import copy
import functools
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any, List

from babydragon.memory.indexes.memory_index import MemoryIndex


class RateLimiter:
    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.interval = 60 / calls_per_minute
        self.lock = Lock()
        self.last_call_time = None

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.lock:
                if self.last_call_time is not None:
                    time_since_last_call = time.time() - self.last_call_time
                    if time_since_last_call < self.interval:
                        time_to_wait = self.interval - time_since_last_call
                        print(
                            f"RateLimiter: Waiting for {time_to_wait:.2f} seconds before next call."
                        )
                        time.sleep(time_to_wait)
                    else:
                        print(
                            f"RateLimiter: No wait required, time since last call: {time_since_last_call:.2f} seconds."
                        )
                else:
                    print("RateLimiter: This is the first call, no wait required.")
                self.last_call_time = time.time()
            return func(*args, **kwargs)

        return wrapper


class RateLimitedThreadPoolExecutor(ThreadPoolExecutor):
    def __init__(self, max_workers=None, *args, **kwargs):
        super().__init__(max_workers)
        self.rate_limiter = RateLimiter(kwargs.get("calls_per_minute", 20))

    def submit(self, fn, *args, **kwargs):
        rate_limited_fn = self.rate_limiter(fn)
        return super().submit(rate_limited_fn, *args, **kwargs)


class BaseTask:
    def __init__(
        self,
        index: MemoryIndex,
        path: List[List[int]],
        max_workers: int = 1,
        task_id: str = "task",
        calls_per_minute: int = 20,
    ):
        self.task_id = task_id
        self.index = index
        self.path = path
        self.results = []
        self.max_workers = max_workers
        self.parallel = True if max_workers > 1 else False
        self.rate_limiter = RateLimiter(calls_per_minute)

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

        with RateLimitedThreadPoolExecutor(
            max_workers=self.max_workers,
            calls_per_minute=self.rate_limiter.calls_per_minute,
        ) as executor:
            futures = []
            print(f"Executing task {self.task_id} using {self.max_workers} workers.")

            for i, sub_path in enumerate(self.path):
                if i < len(self.results):
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
                    self.results.append(sub_task_result)
                    self._save_results_to_file()
                    save_end_time = time.time()
                    print(
                        f"Sub-task {i} results saved in {save_end_time - save_start_time:.2f} seconds."
                    )
                except KeyboardInterrupt:
                    print("Keyboard interrupt detected, stopping task execution.")
                    executor.shutdown(wait=False)
                    break

        print("Task execution completed.")
