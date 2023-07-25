from typing import Any, Optional, Union

from numpy import source
from babydragon.models.generators.polars_batch_generator_models import StatusTrackerModel, OpenaiRequestModel
import polars as pl
import logging
import json
import aiohttp  # for making API calls concurrently
import asyncio  # for running API calls concurrently
import tiktoken  # for counting tokens
import time  # for sleeping after rate limit is hit
from time import time as now
import os

class PolarsGenerator:

    def __init__(
        self,
        input_df: Union[pl.DataFrame,str] = 'noinput',
        name: str = "summarizer",
        tokenizer: Optional[Any] = None,
        save_path: str = 'batch_generator',
        process_objects_number:int = 17,
        logging_level: int = 10,
    ) -> None:

        if isinstance(input_df, pl.DataFrame):
            self.load_path = f"{save_path}/{name}.ndjson" 
            input_df.write_ndjson(self.load_path)
        elif input_df == 'noinput':
            raise TypeError('Constructor requires either a pl.Dataframe or a path to a ndjson')
        elif isinstance(input_df, str):
            self.load_path = input_df
        else:
            raise TypeError('Constructor requires either a pl.Dataframe or a path to a ndjson')
        
        # Settings

        self.name = name
        self.process_objects_number = process_objects_number
        self.max_power_process = process_objects_number*10000
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


        # Status Tracker Logs - polars
        self.st_model = StatusTrackerModel(name=name)
        self.st: pl.DataFrame = pl.DataFrame(self.st_model.model_dump())
        
        # loads the frame 
        self.frame = pl.read_ndjson(self.load_path)

        # queues
        self.requests_queue = asyncio.Queue()
        self.retries_queue = asyncio.Queue()

        # settings
        self.available_request_capacity = 3500
        self.available_token_capacity = 180000
        self.reset_time_token_capacity = 0.0
        self.reset_time_request_capacity = 0.0

        self.time_of_last_rate_limit_error = 0.0

        # api authentication
        self.api_key =  os.getenv("OPENAI_API_KEY")
        self.request_header = {"Authorization": f"Bearer {self.api_key}"}
    
        logging.debug(f"Initialization complete.")

    def enqueue_objects(self):
        id = 0
        with open(self.load_path, 'r') as jsonl_file:
            for line in jsonl_file: 
                line = line.strip()
                if not line:
                    continue
                json_obj = json.loads(line)
                request = OpenaiRequest(**json_obj)
                self.requests_queue.put_nowait((id,request))
                id += 1
                self.len_queue = self.requests_queue.qsize()


    async def process_objects(self):

        while True:

                    next_request = None
                    source_queue = None
              
                    if next_request is None:
                        try:
                                next_request = self.retries_queue.get_nowait()
                                source_queue = self.retries_queue
                                self.st.replace("num_tasks_started", self.st['num_tasks_started']+1)
                                logging.debug(f"Retrying request with id: {next_request[0]}")
                        except asyncio.QueueEmpty:
                                pass
                    if next_request is None:
                        try:
                                next_request = self.requests_queue.get_nowait()
                                source_queue = self.requests_queue
                                self.st.replace("num_tasks_started", self.st['num_tasks_started']+1)
                                logging.info(f'Next request is {next_request[0]} of {self.len_queue}') 
                                
                        except asyncio.QueueEmpty:
                                logging.info("Exiting the loop")
                                break

                    if next_request is not None:

                        next_request_tokens = next_request[1].total_tokens

                        if (
                            self.available_request_capacity >= 1
                            and self.available_token_capacity >= next_request_tokens
                        ):
                            self.available_request_capacity -=1
                            self.available_token_capacity =  self.available_token_capacity-next_request_tokens 

                            try:
                                logging.info(f"Calling Api for {next_request[0]}...")
                                start_time = int(time.time())
                                async with aiohttp.ClientSession() as session:
                                    async with session.post(
                                    url=next_request[1].url, headers=self.request_header, json=next_request[1].body
                                    ) as response:
                                        headers = response.headers
                                        response = await response.json()
                                if "error" in response:
                                    logging.warning(
                                        f"Request {next_request[0]} failed with error {response['error']['message']}"
                                    )
                                    
                                    if "Rate limit" in response["error"].get("message", ""):
                                        self.time_of_last_rate_limit_error = now()
                                        self.available_token_capacity =  0 
                                        self.available_request_capacity +=1
                                        self.st.replace("num_rate_limit_errors", self.st['num_rate_limit_errors']+1)
                                        self.retries_queue.put_nowait(next_request)
                                        output = {
                                        'type': 'RateLimit',
                                        'time': now(),
                                        'response': str(response)
                                        }

                                        json_string = json.dumps(output)

                                        with open(self.error_path, "a") as f:

                                            f.write(json_string + "\n")
                                        

                                    elif "currently overloaded" in response["error"].get("message", ""):
                                        self.available_request_capacity +=1
                                        self.available_token_capacity =  self.available_token_capacity+next_request_tokens
                                        self.st.replace("num_overloaded_errors", self.st['num_overloaded_errors']+1)
                                        self.retries_queue.put_nowait(next_request)
 
                                    else:
                                        self.available_request_capacity +=1
                                        self.available_token_capacity =  self.available_token_capacity+next_request_tokens 
                                        self.st.replace("num_api_errors", self.st['num_api_errors']+1)
                                        json_string = json.dumps(response)
                                        with open(self.error_path, "a") as f:
                                            f.write(json_string + "\n") 
                                else:
                                    output = ''
                                    remaining_token_capacity = int(headers['x-ratelimit-remaining-requests'])
                                    self.available_token_capacity = int(headers['x-ratelimit-remaining-tokens'])
                                    self.available_request_capacity = remaining_token_capacity
                                    reset_time_token_capacity = headers['x-ratelimit-reset-tokens']
                                    logging.info(f"From Headers: Available_token_capacity changed to {self.available_token_capacity} for request with id {next_request[0]}")
                                    if "ms" in reset_time_token_capacity:  
                                        
                                        self.reset_time_token_capacity =  float(reset_time_token_capacity.replace("ms", "")) / 1000.0
                                    elif "s" in reset_time_token_capacity:
                                        
                                        self.reset_time_token_capacity = float(reset_time_token_capacity.replace("s", ""))
                                    else:
                                        self.reset_time_token_capacity = 0.01


                                    total_tokens = response['usage']['total_tokens']
                                    if next_request[1].request_type == 'chat':
                                        output = {
                                            'id': next_request[0],
                                            'start_time': start_time,
                                            'output': response['choices'][0]['message']['content'],
                                            'prompt_tokens': response['usage']['prompt_tokens'],
                                            'completion_tokens': response['usage']['completion_tokens'],
                                            'total_tokens': total_tokens,
                                            'end_time': response['created'],
                                            'remaining_token_capacity': int(headers['x-ratelimit-remaining-tokens']),
                                        }   
                                    elif next_request[1].request_type == 'embedding':

                                        output = {
                                            'id': next_request[0],
                                            'start_time': start_time,
                                            'output': response['data'][0]['embedding'],
                                            'prompt_tokens': response['usage']['prompt_tokens'],
                                            'total_tokens': total_tokens,
                                            'end_time': now(),
                                            'remaining_token_capacity': int(headers['x-ratelimit-remaining-tokens']),
                                        }

                                    json_string = json.dumps(output)
                                     
                                    with open(self.save_path, "a") as f:
                                        f.write(json_string + "\n")
   
                                    if self.available_token_capacity < 3000:
                                        await asyncio.sleep(self.reset_time_token_capacity/self.max_power_process)
                                    else:
                                        max_power_process = (self.max_power_process+1)
                                        if max_power_process >= self.process_objects_number*10000:
                                            self.max_power_process = self.process_objects_number*10000
                                        else:
                                            self.max_power_process = max_power_process
                                    logging.info(f"Max power process is :{self.max_power_process}")

                                                                                                                                              
                            except Exception as e:
                                logging.warning(f"Request {next_request[0]} failed with Exception {e}")
                                self.st.replace("num_other_errors", self.st['num_other_errors']+1)
                                self.available_token_capacity = self.available_token_capacity+next_request_tokens
                                self.available_request_capacity = self.available_request_capacity+1
                                output = {
                                    'type': 'other',
                                    'time': now(),
                                    'response': str(e)
                                }

                                json_string = json.dumps(output)
                                with open(self.error_path, "a") as f:

                                    f.write(json_string + "\n")

                            finally:
                                if source_queue is not None:
                                    source_queue.task_done()



                        else:
                            seconds_since_rate_limit_error = (now() - self.time_of_last_rate_limit_error)
                            if seconds_since_rate_limit_error < 3 :
                                self.max_power_process = self.process_objects_number
                                logging.info(f"Max power process decreased to :{self.max_power_process}")
                                time_to_wait = (self.reset_time_token_capacity - seconds_since_rate_limit_error)
                                logging.warn(f"Pausing to reset_time_token_capacity = {time_to_wait/self.max_power_process}")
                                await asyncio.sleep(time_to_wait/self.max_power_process)
                                self.retries_queue.put_nowait(next_request)
                                self.time_of_last_rate_limit_error = 0.0
                                self.available_token_capacity = 180000/self.max_power_process
                            else:
                                await asyncio.sleep(self.reset_time_token_capacity/self.max_power_process)
                                self.available_token_capacity = self.available_token_capacity + next_request_tokens
                                self.retries_queue.put_nowait(next_request)

                            
                            if source_queue is not None:
                                    source_queue.task_done()

                            
                    



    async def main(self):
        logging.debug(f"Entering main loop")
        self.enqueue_objects()
        self.consumers = [asyncio.create_task(self.process_objects()) for _ in range(self.process_objects_number)]
        await self.requests_queue.join()
        await self.retries_queue.join()
        for consumer in self.consumers:
            consumer.cancel()
        logging.info(f"""Parallel processing complete. Results saved to {self.save_path}""")
        if self.st['num_rate_limit_errors'][0] > 0:
            logging.warning(f"{self.st['num_rate_limit_errors'][0]} rate limit errors received. Consider running at a lower rate.")
        
        self.st.write_ndjson(self.log_path)
        print(self.st)

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
                max_tokens=parameters.get('max_tokens',1000),
                presence_penalty=parameters.get('presence_penalty'),
                frequency_penalty=parameters.get('frequency_penalty'),
                logit_bias=parameters.get('logit_bias'),
                user=parameters.get('user')
        )

        self.body = {k: v for k, v in self.body.model_dump().items() if v is not None}
        if self.request_type == 'embedding':
            keys_to_remove = ["n", "max_tokens"]
            self.body = {k: v for k, v in self.body.items() if k not in keys_to_remove}

        """Count the number of tokens in the request. Only supports completion and embedding requests."""
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

        if self.request_type == 'chat':

            completion_tokens = self.body['max_tokens']
            num_tokens = 0
            for message in self.body["messages"]:
                    num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                    for key, value in message.items():
                        num_tokens += len(encoding.encode(value))
                        if key == "name":  # if there's a name, the role is omitted
                            num_tokens -= 1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            self.total_tokens = num_tokens + completion_tokens

        elif self.request_type == "embeddings":
            input = self.body["input"]
            if isinstance(input, str):  # single input
                self.total_tokens = len(encoding.encode(input))
            elif isinstance(input, list):  # multiple inputs
                self.total_tokens = sum([len(encoding.encode(i)) for i in input])

        self.max_tokens_per_minute = 0
        self.max_requests_per_minute = 0
        self.url = ''

        match self.body['model']: # pyright: ignore
            case 'gpt-3.5-turbo':

                    self.max_requests_per_minute = 3500
                    self.max_tokens_per_minute= 90000
                    self.url = 'https://api.openai.com/v1/chat/completions'

            case 'gpt-3.5-turbo-16k':

                    self.max_requests_per_minute = 3500
                    self.max_tokens_per_minute= 180000
                    self.url = 'https://api.openai.com/v1/chat/completions'

            case 'gpt-4':

                    self.max_requests_per_minute = 3500
                    self.max_tokens_per_minute= 90000
                    self.url = 'https://api.openai.com/v1/chat/completions'


            case 'text-embedding-ada-002':

                    self.max_requests_per_minute = 3000
                    self.url = 'https://api.openai.com/v1/embeddings'