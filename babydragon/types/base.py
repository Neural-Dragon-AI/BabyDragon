from pydantic import BaseModel, Field, field_validator
from typing import Union, List, Optional, Dict
import datetime
import uuid
import tiktoken
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")


# Base class for all bd_types
class BDType(BaseModel):
    source: str = Field("babydragon", description="The source of the data.")
    timestamp: Optional[datetime.datetime] = Field(None, description="When the data was collected or created. If not provided, the current time is used.")
    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="Unique identifier of the data.")
    data_name: Optional[str] = Field(None, description="Name of the data.")
    elements_name: Optional[List[str]] = Field(None, description="Names of the elements if the data is a list.")

    @field_validator("timestamp")
    def set_timestamp(cls, v):
        return v or datetime.datetime.now()

    @field_validator("id")
    def set_id(cls, values, **kwargs):
        if "id" not in values:
            values["id"] = uuid.uuid4()
        return values