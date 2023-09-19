from typing import  List, Optional

from pydantic import ConfigDict, BaseModel

class CodeFramePydantic(BaseModel):
    df_path: str
    context_columns: List
    embeddable_columns: List
    embedding_columns: List
    name: str
    save_path: Optional[str]
    save_dir: str
    markdown: str
    model_config = ConfigDict(arbitrary_types_allowed=True)


class MemoryFramePydantic(BaseModel):
    df_path: str
    context_columns: List[str]
    embeddable_columns: List[str]
    embedding_columns: List[str]
    time_series_columns: List[str]
    name: str
    save_path: Optional[str]
    save_dir: str
    markdown: str
    model_config = ConfigDict(arbitrary_types_allowed=True)