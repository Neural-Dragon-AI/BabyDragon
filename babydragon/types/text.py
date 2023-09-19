from typing import List

import tiktoken
from pydantic import Field, field_validator

from babydragon.types.base import BDType

tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")


class NaturalLanguageSingle(BDType):
    text: str = Field(
        ...,
        description="The natural language text. It should be less than or equal to `max_tokens` in length when tokenized.",
    )
    max_tokens: int = Field(
        8000,
        description="The maximum allowed length of the text in tokens. The default value is 8000.",
    )

    @field_validator("text")
    def validate_text(cls, v, info):
        try:
            # Tokenize the text and get the token count
            token_count = len(tokenizer.encode(v))
        except Exception as e:
            raise ValueError("Failed to tokenize text.") from e

        # Get max_tokens from info.data, if not available, default to 8000
        max_tokens = info.data.get("max_tokens", 8000)

        if token_count > max_tokens:
            raise ValueError(f"Text is longer than {max_tokens} tokens.")

        return v


class NaturalLanguageList(BDType):
    texts: List[NaturalLanguageSingle] = Field(
        ...,
        description="A list of `NaturalLanguageSingle` objects. Each object should pass the validation requirements of the `NaturalLanguageSingle` class.",
    )
