from typing import Any, Optional, Union, List, Dict
import polars as pl
import logging
import concurrent.futures
import json
import aiohttp  # for making API calls concurrently
import asyncio  # for running API calls concurrently
import tiktoken  # for counting tokens
import time  # for sleeping after rate limit is hit
from datetime import datetime
from dataclasses import dataclass, field  # for storing API inputs, outputs, and metadata
import os

class BatchGenerator:

    def __init__(
        self,
        input_df: Union[pl.DataFrame,str] = 'noinput',
        task: str = 'chat',
        name: str = "summarizer",
        tokenizer: Optional[Any] = None,
        save_path: str = 'batch_generator',
        logging_level: int = 10,
    ) -> None:

        if isinstance(input_df, pl.DataFrame):
            self.load_path = f"{save_path}/{name}.ndjson" ## pyright: ignore
            input_df.write_ndjson(self.load_path)
        elif input_df == 'noinput':
            raise TypeError('Constructor requires either a pl.Dataframe or a path to a ndjson')
        else:
            self.load_path = input_df

        self.name = name
        self.task = task
        self.save_path = save_path
        self.logging_level = logging_level    

        if tokenizer is None:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        else:
            self.tokenizer = tokenizer

        # debug loads the frame in advance for usefull checks 
        self.frame = pl.read_ndjson(self.load_path)

        # queues
        self.requests_queue = asyncio.Queue()
        self.retries_queue = asyncio.Queue()
        self.errors_queue = asyncio.Queue()

        # constants
        self.max_attempts = 5
        self.seconds_to_pause_after_rate_limit_error = 15
        self.seconds_to_sleep_each_loop = 0.001  # 1 ms limits max throughput to 1,000 requests per second

        # initialize logging
        logging.basicConfig(level=logging_level)
        logging.debug(f"Logging initialized at level {logging_level}")
        
        # initialize available capacity counts
        self.available_request_capacity = 1500
        self.available_token_capacity = 6250000
        self.last_update_time = time.time()

        # api authentication
        self.api_key =  os.getenv("OPENAI_API_KEY")
        self.request_header = {"Authorization": f"Bearer {self.api_key}"}
    
        @dataclass
        class StatusTracker:
            num_tasks_started: int = 0
            num_tasks_in_progress: int = 0  # script ends when this reaches 0
            num_tasks_succeeded: int = 0
            num_tasks_failed: int = 0
            num_rate_limit_errors: int = 0
            num_api_errors: int = 0  # excluding rate limit errors, counted above
            num_other_errors: int = 0
            time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits

        self.status_tracker = StatusTracker()

        self.finished = False  # after file is empty, we'll skip reading it
        logging.debug(f"Initialization complete.")

    def enqueue_objects(self):
        with open(self.load_path, 'r') as jsonl_file:
            for line in jsonl_file:
                line = line.strip()
                if not line:
                    continue
                json_obj = json.loads(line)                
                self.requests_queue.put_nowait(json_obj)  


    async def process_objects(self, queue):
        next_request = None  # variable to hold the next request to call
        while True:
                    # get next request (if one is not already waiting for capacity)
                    if not self.retries_queue.empty():
                            next_request = self.retries_queue.get_nowait()
                            logging.debug(f"Retrying request: {next_request}")
                    elif not self.requests_queue.empty():
                            try:
                                logging.debug(f"Trying to retrieve next request")
                                next_request = self.requests_queue.get_nowait()     
                                logging.info(f'Next request is {next_request}')
                                if next_request.respect_token_limit:
                                    self.status_tracker.num_tasks_started += 1
                                    self.status_tracker.num_tasks_in_progress += 1
                                    logging.debug(f"Reading request: {next_request}")
                                else:
                                    self.errors_queue.put_nowait(next_request)
                            except StopIteration:
                                # if file runs out, set flag to stop reading it
                                logging.debug("Read file exhausted")
                                self.finished = True

                    # update available capacity
                    current_time = time.time()
                    seconds_since_update = current_time - self.last_update_time
                    self.available_request_capacity = min(
                    self.available_request_capacity + next_request.max_requests_per_minute * seconds_since_update / 60.0, ## pyright: ignore
                        next_request.max_requests_per_minute, ## pyright: ignore
                    )
                    self.available_token_capacity = min(
                        self.available_token_capacity + next_request.max_tokens_per_minute * seconds_since_update / 60.0, ## pyright: ignore
                        next_request.max_tokens_per_minute,  ## pyright: ignore
                    )
                    self.last_update_time = current_time

                    # if enough capacity available, call API
                    if next_request:
                        next_request_tokens = next_request.total_tokens
                        if (
                            self.available_request_capacity >= 1
                            and self.available_token_capacity >= next_request_tokens
                        ):
                            # update counters
                            self.available_request_capacity -= 1
                            self.available_token_capacity -= next_request_tokens
                            

                            # call API
                            try:
                                logging.info(f"Calling Api for {next_request.body}")
                                async with aiohttp.ClientSession() as session:
                                    async with session.post(
                                    url=next_request.url, headers=self.request_header, json=next_request.body
                                    ) as response:
                                        response = await response.json()
                                if "error" in response:
                                    logging.warning(
                                        """ f"Request {self.task_id} failed with error {response['error']}" """
                                    )
                                    self.status_tracker.num_api_errors += 1
                                    if "Rate limit" in response["error"].get("message", ""):
                                        self.status_tracker.time_of_last_rate_limit_error = time.time() # pyright: ignore 
                                        self.status_tracker.num_rate_limit_errors += 1
                                        self.status_tracker.num_api_errors -= 1  # rate limit errors are counted separately
                                        self.retries_queue.put_nowait(next_request)
                                    else:
                                        self.errors_queue.put_nowait(next_request)
                                else:
                                        print(response)
                                        

                            except Exception as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
                                logging.warning(f"Request {next_request} failed with Exception {e}")
                                self.status_tracker.num_other_errors += 1
                            queue.task_done() 
                            next_request = None  # reset next_request to empty

                    # if all tasks are finished, break
                    if self.status_tracker.num_tasks_in_progress == 0:
                        break

                    # main loop sleeps briefly so concurrent tasks can run
                    await asyncio.sleep(self.seconds_to_sleep_each_loop)

                    # if a rate limit error was hit recently, pause to cool down
                    seconds_since_rate_limit_error = (time.time() - self.status_tracker.time_of_last_rate_limit_error)
                    if seconds_since_rate_limit_error < self.seconds_to_pause_after_rate_limit_error:
                        remaining_seconds_to_pause = (self.seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error)
                        await asyncio.sleep(remaining_seconds_to_pause)
                        # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
                        logging.warn(f"Pausing to cool down until {time.ctime(self.status_tracker.time_of_last_rate_limit_error + self.seconds_to_pause_after_rate_limit_error)}")

        # after finishing, log final status
        logging.info(f"""Parallel processing complete. Results saved to {self.save_path}""")
        if self.status_tracker.num_tasks_failed > 0:
            logging.warning(f"{self.status_tracker.num_tasks_failed} / {self.status_tracker.num_tasks_started} requests failed. Errors logged to {self.save_path}.")
        if self.status_tracker.num_rate_limit_errors > 0:
            logging.warning(f"{self.status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate.")



    async def main(self):
        logging.debug(f"Entering main loop")
        self.enqueue_objects()
        consumers = [asyncio.create_task(self.process_objects(self.requests_queue)) for _ in range(5)]
        await self.requests_queue.join()
        await self.retries_queue.join()
        for consumer in consumers:
            consumer.cancel()

    def execute(self):
        asyncio.run(self.main())


class ApiRequest:
    def __init__(
        self, 
        **parameters
        ) -> None:

        if ('input' in parameters) or ('input' in parameters.get('body', {})):  
            self.request_type = 'embedding'
        else:
            self.request_type = 'chat'
        

        if parameters.get('body') is not None:
            self.body = parameters.get('body')

        else:
            self.body = {
                "model": parameters.get('model'),
                "input": parameters.get('input'),
                "messages": parameters.get('messages'),
                "function": parameters.get('function'),
                "function_call": parameters.get('function_call'),
                "temperature": parameters.get('temperature'),
                "top_p": parameters.get('top_p'),
                "n": parameters.get('n'),
                "stream": parameters.get('stream'),
                "stop": parameters.get('stop'),
                "max_tokens": parameters.get('max_tokens'),
                "presence_penalty": parameters.get('presence_penalty'),
                "frequency_penalty": parameters.get('frequency_penalty'),
                "logit_bias": parameters.get('logit_bias'),
                "user": parameters.get('user')
            }

        # Remove keys with None values
        if self.body is not None:
            self.body = {k: v for k, v in self.body.items() if v is not None}

        """Count the number of tokens in the request. Only supports chat and embedding requests."""
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

        # if completions request, tokens = prompt + n * max_tokens
        if self.request_type == 'chat':
            max_tokens = self.body['max_tokens'] # pyright: ignore
            n = self.body['n'] # pyright: ignore
            completion_tokens = n * max_tokens # pyright: ignore
            for message in self.body["messages"]: # pyright: ignore
                self.total_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    self.total_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        self.total_tokens -= 1  # role is always required and always 1 token
                    self.total_tokens += 2  # every reply is primed with <im_start>assistant
                self.total_tokens =  self.total_tokens + completion_tokens
            
        # if embeddings request, tokens = input tokens
        elif self.request_type == "embedding":
            if isinstance(self.body['input'], str):  # single input # pyright: ignore
                self.total_tokens = len(encoding.encode(self.body["input"]))  # pyright: ignore
            elif isinstance(input, list):  # multiple inputs
                self.total_tokens = sum([len(encoding.encode(i)) for i in self.body["input"]]) # pyright: ignore
            else:
                raise TypeError('Expecting either string or list of strings for "inputs" field in embedding request')

        match self.body['model']: # pyright: ignore
            case 'gpt-3.5-turbo':
                if self.total_tokens < 4000:
                    self.respect_token_limit = True
                    self.max_requests_per_minute = 3000
                    self.max_tokens_per_minute= 1000000
                    self.url = 'https://api.openai.com/v1/chat/completions'
                else:
                    self.respect_token_limit = False
            case 'gpt-4':
                if self.total_tokens < 16000:
                    self.respect_token_limit = True
                    self.max_requests_per_minute = 3000
                    self.max_tokens_per_minute= 1000000
                    self.url = 'https://api.openai.com/v1/chat/completions'
                else:
                    self.respect_token_limit = False

            case 'ext-embedding-ada-002':
                if self.total_tokens < 4000:
                    self.respect_token_limit = True
                    self.max_tokens_per_minute = 1000000
                    self.max_requests_per_minute = 3000
                    self.url = 'https://api.openai.com/v1/embeddings'
                else:
                    self.respect_token_limit = False                

                


