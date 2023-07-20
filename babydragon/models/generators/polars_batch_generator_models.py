from pydantic import BaseModel
from typing import Any, Union, List, Mapping


class StatusTrackerModel(BaseModel):
    name: str

    num_rate_limit_errors: int = 0
    num_overloaded_errors: int = 0
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
