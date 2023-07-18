from pydantic import BaseModel
from time import time as now
from typing import Any, Union, List, Mapping


class StatusTrackerModel(BaseModel):
    name: str
    max_attempts: int = 5
    seconds_to_pause_after_rate_limit_error: int = 10
    seconds_to_sleep_each_loop: float  = 0.001
    available_request_capacity: int = 1500
    available_token_capacity: int = 625000
    last_update_time: float = now()
    num_rate_limit_errors: int = 0
    num_overloaded_errors: int = 0
    time_of_last_rate_limit_error: float = 0.0
    num_tasks_failed: int = 0
    num_tasks_started: int = 0
    num_api_errors: int = 0
    num_other_errors: int = 0 



class OpenaiRequestModel(BaseModel):
    model: str
    input: Union[str,List,None]
    messages: Union[List,None]
    function: Union[List,None]
    function_call: Union[str,Any,None]
    temperature: Union[float,None]
    top_p: Union[float,None]
    n: Union[int,None]
    stream: Union[bool,None]
    stop: Union[str,List,None]
    max_tokens: Union[int,None]
    presence_penalty: Union[float,None]
    frequency_penalty: Union[float,None]
    logit_bias: Union[Mapping,None]
    user: Union[str,None]
