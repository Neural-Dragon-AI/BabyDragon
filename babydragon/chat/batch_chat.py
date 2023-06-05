from babydragon.models.batch_request_openai import api_endpoint_from_url, num_tokens_consumed_from_request, task_id_generator_function, APIRequest, StatusTracker
import asyncio
import time
from typing import Dict
from babydragon.chat.base_chat import BaseChat
import logging  # for logging rate limit warnings and other messages
from typing import List, Tuple

# Set the variables


async def process_chat_requests(
    request_data: List[Dict],
    save_filepath: str,
    request_url: str,
    api_key: str,
    max_requests_per_minute: float,
    max_tokens_per_minute: float,
    token_encoding_name: str,
    max_attempts: int,
    logging_level: int,
):
    """Processes chat requests in parallel, throttling to stay under rate limits."""
    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = 0.001  # 1 ms limits max throughput to 1,000 requests per second

    # initialize logging
    logging.basicConfig(level=logging_level)
    logging.debug(f"Logging initialized at level {logging_level}")

    # infer API endpoint and construct request header
    api_endpoint = api_endpoint_from_url(request_url)
    request_header = {"Authorization": f"Bearer {api_key}"}

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = task_id_generator_function()  # generates integer IDs of 1, 2, 3, ...
    status_tracker = StatusTracker()  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # initialize flags
    file_not_finished = True  # after file is empty, we'll skip reading it
    logging.debug(f"Initialization complete.")

    # `requests` will provide requests one at a time
    requests = iter(request_data)
    logging.debug(f"List initialized. Entering main loop")

    while True:
        # get next request (if one is not already waiting for capacity)
        if next_request is None:
            if not queue_of_requests_to_retry.empty():
                next_request = queue_of_requests_to_retry.get_nowait()
                logging.debug(f"Retrying request {next_request.task_id}: {next_request}")
            elif file_not_finished:
                try:
                    # get new request
                    request_json = next(requests)
                    next_request = APIRequest(
                        task_id=next(task_id_generator),
                        request_json=request_json,
                        token_consumption=num_tokens_consumed_from_request(request_json, api_endpoint, token_encoding_name),
                        attempts_left=max_attempts,
                        metadata=request_json.pop("metadata", None)
                    )
                    status_tracker.num_tasks_started += 1
                    status_tracker.num_tasks_in_progress += 1
                    logging.debug(f"Reading request {next_request.task_id}: {next_request}")
                except StopIteration:
                    # if file runs out, set flag to stop reading it
                    logging.debug("Read file exhausted")
                    file_not_finished = False

        # update available capacity
        current_time = time.time()
        seconds_since_update = current_time - last_update_time
        available_request_capacity = min(
            available_request_capacity + max_requests_per_minute * seconds_since_update / 60.0,
            max_requests_per_minute,
        )
        available_token_capacity = min(
            available_token_capacity + max_tokens_per_minute * seconds_since_update / 60.0,
            max_tokens_per_minute,
        )
        last_update_time = current_time

        # if enough capacity available, call API
        if next_request:
            next_request_tokens = next_request.token_consumption
            if (
                available_request_capacity >= 1
                and available_token_capacity >= next_request_tokens
            ):
                # update counters
                available_request_capacity -= 1
                available_token_capacity -= next_request_tokens
                next_request.attempts_left -= 1

                # call API
                asyncio.create_task(
                    next_request.call_api(
                        request_url=request_url,
                        request_header=request_header,
                        retry_queue=queue_of_requests_to_retry,
                        save_filepath=save_filepath,
                        status_tracker=status_tracker,
                    )
                )
                next_request = None  # reset next_request to empty

        # if all tasks are finished, break
        if status_tracker.num_tasks_in_progress == 0:
            break

        # main loop sleeps briefly so concurrent tasks can run
        await asyncio.sleep(seconds_to_sleep_each_loop)

        # if a rate limit error was hit recently, pause to cool down
        seconds_since_rate_limit_error = (time.time() - status_tracker.time_of_last_rate_limit_error)
        if seconds_since_rate_limit_error < seconds_to_pause_after_rate_limit_error:
            remaining_seconds_to_pause = (seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error)
            await asyncio.sleep(remaining_seconds_to_pause)
            # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
            logging.warn(f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}")

    # after finishing, log final status
    logging.info(f"""Parallel processing complete. Results saved to {save_filepath}""")
    if status_tracker.num_tasks_failed > 0:
        logging.warning(f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {save_filepath}.")
    if status_tracker.num_rate_limit_errors > 0:
        logging.warning(f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate.")


class BatchChat(BaseChat):
    def __init__(self, model: str = "gpt-3.5-turbo", max_output_tokens: int = 200, save_filepath: str = None, max_attempts: int = 5, api_key: str = None ):
        #super init
        super().__init__(model=model, max_output_tokens=max_output_tokens)
        # Set the variables
        if save_filepath is None:
            self.save_filepath = "results.jsonl"  # Default filename if not using input file
        else:
            self.save_filepath = save_filepath
        self.request_url = "https://api.openai.com/v1/chat/completions" # URL for the chat endpoint
        self.max_requests_per_minute = 3_000 * 0.5
        self.max_tokens_per_minute = 250_000 * 0.5
        self.token_encoding_name = "cl100k_base"
        self.api_key = api_key
        self.max_attempts = max_attempts
        self.logging_level = logging.INFO

    async def batch_query(self, messages: List[str], system_prompts:List[str]) -> List[str]:
        request_data = []
        for message, system_prompt in zip(messages, system_prompts):
            prompt, _ = self.prompt_func(message)
            print(prompt)
            request_data.append({
                "model": self.model,
                "max_tokens": self.max_output_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ]
            })
        responses = await process_chat_requests(request_data=request_data, request_url=self.request_url, api_key=self.api_key, save_filepath=self.save_filepath, max_requests_per_minute=self.max_requests_per_minute, max_tokens_per_minute=self.max_tokens_per_minute, token_encoding_name=self.token_encoding_name, max_attempts=self.max_attempts, logging_level=self.logging_level)
        return responses
