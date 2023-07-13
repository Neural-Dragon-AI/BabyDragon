from typing import Any, Optional, Union, List, Dict
from babydragon.models.generators.polars_batch_generator_models import StatusTrackerModel, OpenaiRequestModel
import polars as pl
import logging
import json
import aiohttp  # for making API calls concurrently
import asyncio  # for running API calls concurrently
import tiktoken  # for counting tokens
import time  # for sleeping after rate limit is hit
from datetime import datetime
import os

class PolarsGenerator:

    def __init__(
        self,
        input_df: Union[pl.DataFrame,str] = 'noinput',
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
        elif isinstance(input_df, str):
            self.load_path = input_df
        else:
            raise TypeError('Constructor requires either a pl.Dataframe or a path to a ndjson')
        

        # Settings

        self.name = name
        self.save_path =f"{save_path}/{self.name}_output.ndjson"
        self.error_path = f"{save_path}/{self.name}_errors.ndjson"
        self.log_path = f"{save_path}/{self.name}_log.ndjson"

        self.logging_level = logging_level  
        
        if tokenizer is None:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        else:
            self.tokenizer = tokenizer


        logging.basicConfig(level=logging_level)
        logging.debug(f"Logging initialized at level {logging_level}")


        # Status Tracker - polars
        self.st_model = StatusTrackerModel(name=name)
        self.st: pl.DataFrame = pl.DataFrame(self.st_model.model_dump())
        
        # loads the frame in advance for usefull checks 
        self.frame = pl.read_ndjson(self.load_path)

        # queues
        self.requests_queue = asyncio.Queue()
        self.retries_queue = asyncio.Queue()
        self.errors_queue = asyncio.Queue()


        # api authentication
        self.api_key =  os.getenv("OPENAI_API_KEY")
        self.request_header = {"Authorization": f"Bearer {self.api_key}"}
    
        logging.debug(f"Initialization complete.")

    def enqueue_objects(self):
        id = 0
        with open(self.load_path, 'r') as jsonl_file:
            for line in jsonl_file:
                id += 1
                line = line.strip()
                if not line:
                    continue
                json_obj = json.loads(line)
                request = OpenaiRequest(**json_obj)
                self.requests_queue.put_nowait((id,request))
                self.len_queue = self.requests_queue.qsize()


    async def process_objects(self):
        while True:
                    next_request = None
                    retry = False
                    if not self.retries_queue.empty():
                            next_request = self.retries_queue.get_nowait()
                            logging.debug(f"Retrying request: {next_request[0]}")
                            retry = True
                    elif not self.requests_queue.empty():                       
                            logging.debug(f"Trying to retrieve next request")
                            next_request = self.requests_queue.get_nowait()    
                            logging.info(f'Next request is {next_request[0]} of {self.len_queue}')
                            logging.info(f'Respects token limit? {next_request[1].respect_token_limit}')
                            if next_request[1].respect_token_limit:
                                logging.debug(f"Reading request: {next_request[0]}")
                            else:
                                self.errors_queue.put_nowait(next_request)
                    else:
                        break

                    current_time = time.time()
                    seconds_since_update = current_time - self.st['last_update_time'][0]

                    if next_request is not None:

                        available_request_capacity = pl.Series([min([
                            self.st['available_request_capacity'][0] + next_request[1].max_requests_per_minute * seconds_since_update / 60.0,
                            next_request[1].max_requests_per_minute
                        ])])
                        self.st.replace("available_request_capacity", available_request_capacity)
                        logging.info(f"Task number {next_request[0]} available_request_capacity: {self.st['available_request_capacity'][0]}")


                        available_token_capacity = pl.Series([min([
                            self.st['available_token_capacity'][0] + next_request[1].max_tokens_per_minute * seconds_since_update / 60.0,
                            next_request[1].max_tokens_per_minute
                        ])])
                        self.st.replace("available_token_capacity", available_token_capacity)
                        logging.info(f"Task number {next_request[0]} available_token_capacity: {self.st['available_token_capacity'][0]}")

                        next_request_tokens = next_request[1].total_tokens


                        if (
                            self.st['available_request_capacity'][0] >= 1
                            and self.st['available_token_capacity'][0] >= next_request_tokens
                        ):
                            # update counters
                            self.st.replace("available_request_capacity", available_request_capacity-1)
                            self.st.replace("available_token_capacity", available_token_capacity-next_request_tokens)
                        

                            # call API
                            try:
                                logging.info(f"Calling Api for {next_request[0]}")
                                start_time = time.time()
                                async with aiohttp.ClientSession() as session:
                                    async with session.post(
                                    url=next_request[1].url, headers=self.request_header, json=next_request[1].body
                                    ) as response:
                                        response = await response.json()
                                if "error" in response:
                                    logging.warning(
                                        """ f"Request {self.task_id} failed with error {response['error']}" """
                                    )
                                    self.st.replace("num_api_errors", self.st['num_api_errors']+1)
                                    if "Rate limit" in response["error"].get("message", ""):
                                        self.st.replace("time_of_last_rate_limit_error", pl.Series([time.time()]))
                                        self.st.replace("num_rate_limit_errors", self.st['num_rate_limit_errors']+1)
                                        self.st.replace("num_api_errors", self.st['num_api_errors']-1)

                                        self.retries_queue.put_nowait(next_request)
                                    else:
                                        json_string = json.dumps(response)
                                        with open(self.error_path, "a") as f:
                                            f.write(json_string + "\n") 
                                else:
                                    output = ''
                                    if next_request[1].request_type == 'chat':
                                        output = {
                                            'start_time': start_time,
                                            'output': response['choices'][0]['message']['content'],
                                            'prompt_tokens': response['usage']['prompt_tokens'],
                                            'completion_tokens': response['usage']['completion_tokens'],
                                            'total_tokens': response['usage']['total_tokens'],
                                            'end_time': response['created']
                                        }   
                                    elif next_request[1].request_type == 'embedding':

                                        output = {
                                            'start_time': start_time,
                                            'output': response['data'][0]['embedding'],
                                            'prompt_tokens': response['usage']['prompt_tokens'],
                                            'total_tokens': response['usage']['total_tokens'],
                                            'end_time': time.time()
                                        }

                                    json_string = json.dumps(output)

                                    with open(self.save_path, "a") as f:
                                        f.write(json_string + "\n")

                                    total_tokens = response['usage']['total_tokens']
                                    self.st.replace("available_token_capacity", available_token_capacity+total_tokens)
                                    self.st.replace("available_request_capacity", available_request_capacity+1)
                                    
                                    if retry:
                                        self.retries_queue.task_done()
                                    else:
                                        self.requests_queue.task_done()


                                        
                                        

                            except Exception as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
                                logging.warning(f"Request {next_request[0]} failed with Exception {e}")
                                self.st.replace("num_other_errors", self.st['num_other_errors']+1)
                                self.st.replace("available_token_capacity", available_token_capacity+next_request_tokens)
                                self.st.replace("available_request_capacity", available_request_capacity+1)
                                json_string = json.dumps(str(e))
                                with open(self.error_path, "a") as f:
                                    f.write(json_string + "\n")
                                if retry:
                                    self.retries_queue.task_done()
                                else:
                                    self.requests_queue.task_done()

                    await asyncio.sleep(self.st['seconds_to_sleep_each_loop'][0])

                    # if a rate limit error was hit recently, pause to cool down
                    seconds_since_rate_limit_error = (time.time() - self.st['time_of_last_rate_limit_error'][0])
                    if seconds_since_rate_limit_error < self.st['seconds_to_pause_after_rate_limit_error'][0]:
                        remaining_seconds_to_pause = (self.st['seconds_to_pause_after_rate_limit_error'][0] - seconds_since_rate_limit_error)
                        await asyncio.sleep(remaining_seconds_to_pause)
                        # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
                        logging.warn(f"Pausing to cool down until {time.ctime(self.st['time_of_last_rate_limit_error'][0] + self.st['seconds_to_pause_after_rate_limit_error'][0])}")
                    if retry:
                        self.retries_queue.task_done()
                    else:
                        self.requests_queue.task_done()


        logging.info(f"""Parallel processing complete. Results saved to {self.save_path}""")
        if self.st['num_tasks_failed'][0] > 0:
            logging.warning(f"{self.st['num_tasks_failed'][0]} / {self.st['num_tasks_started'][0]} requests failed. Errors logged to {self.log_path}.")
        if self.st['num_rate_limit_errors'][0] > 0:
            logging.warning(f"{self.st['num_rate_limit_errors'][0]} rate limit errors received. Consider running at a lower rate.")
        
        self.st.write_ndjson(self.log_path)

                    



    async def main(self):
        logging.debug(f"Entering main loop")
        self.enqueue_objects()
        consumers = [asyncio.create_task(self.process_objects()) for _ in range(5)]
        await self.requests_queue.join()
        await self.retries_queue.join()
        for consumer in consumers:
            consumer.cancel()

    def execute(self):
        asyncio.run(self.main())


class OpenaiRequest:
    def __init__(
        self, 
        **parameters
        ) -> None:

        if ('input' in parameters):  
            self.request_type = 'embedding'
        else:
            self.request_type = 'chat'
        
        self.body = OpenaiRequestModel(
                model=parameters.get('model','gpt-3.5-turbo'),
                input=parameters.get('input'),
                messages=parameters.get('messages', [{"role": "system", "content": "You are a helpful assistant."}]) ,
                function=parameters.get('function'),
                function_call=parameters.get('function_call'),
                temperature=parameters.get('temperature'),
                top_p=parameters.get('top_p'),
                n=parameters.get('n',1),
                stream=parameters.get('stream'),
                stop=parameters.get('stop'),
                max_tokens=parameters.get('max_tokens',4000),
                presence_penalty=parameters.get('presence_penalty'),
                frequency_penalty=parameters.get('frequency_penalty'),
                logit_bias=parameters.get('logit_bias'),
                user=parameters.get('user')
        )

        self.body = {k: v for k, v in self.body.model_dump().items() if v is not None}
        if self.request_type == 'embedding':
            keys_to_remove = ["n", "max_tokens"]
            self.body = {k: v for k, v in self.body.items() if k not in keys_to_remove}

        """Count the number of tokens in the request. Only supports chat and embedding requests."""
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.total_tokens = 0
        self.respect_token_limit = False
        self.max_tokens_per_minute = 0
        self.max_requests_per_minute = 0
        self.url = ''

        # if completions request, tokens = prompt + n * max_tokens
        if self.request_type == 'chat':
            max_tokens = self.body['max_tokens']  
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
                if self.total_tokens < 10000:
                    self.respect_token_limit = True
                    self.max_requests_per_minute = 3000
                    self.max_tokens_per_minute= 1000000
                    self.url = 'https://api.openai.com/v1/chat/completions'
                else:
                    self.respect_token_limit = False
                    self.max_requests_per_minute = 0
                    self.max_tokens_per_minute= 0

            case 'gpt-4':
                if self.total_tokens < 16000:
                    self.respect_token_limit = True
                    self.max_requests_per_minute = 3000
                    self.max_tokens_per_minute= 1000000
                    self.url = 'https://api.openai.com/v1/chat/completions'
                else:
                    self.respect_token_limit = False
                    self.max_requests_per_minute = 0
                    self.max_tokens_per_minute= 0


            case 'text-embedding-ada-002':
                if self.total_tokens < 10000:
                    self.respect_token_limit = True
                    self.max_tokens_per_minute = 1000000
                    self.max_requests_per_minute = 3000
                    self.url = 'https://api.openai.com/v1/embeddings'
                else:
                    self.respect_token_limit = False              
                    self.max_requests_per_minute = 0
                    self.max_tokens_per_minute= 0


                


