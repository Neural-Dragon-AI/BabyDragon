from typing import Dict, Generator, List, Optional, Tuple, Union

import polars as pl
from dragon.memory.chat_frame import ChatFrame

from babydragon.chat.base_chat import BaseChat, Prompter
from babydragon.utils.chatml import (
    get_str_from_response,
    mark_question,
)


class MemoryChat(BaseChat):
    def __init__(
        self,
        model: Union[str, None] = None,
        max_output_tokens: int = 200,
        memory_thread_name: str = "memory",
        max_memory: Optional[int] = None,
    ):
        BaseChat.__init__(self, model=model, max_output_tokens=max_output_tokens)
        self.prompter = Prompter()
        self.chat_frame = ChatFrame(name=memory_thread_name, max_memory=max_memory)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text)) + 7

    def identity_prompter(self, message: str) -> Tuple[List[Dict], str]:
        self.add_message_to_thread(role="user", content=message)
        return [mark_question(message)], mark_question(message)

    def add_message_to_thread(self, role: str, content: str):
        message_dict = {"role": role, "content": content}
        self.chat_frame.add_dict_to_thread(message_dict)

    def chat_response(
        self,
        message: List[dict],
        max_tokens: Union[int, None] = None,
        stream: bool = False,
    ) -> Union[Generator, Tuple[Dict, bool]]:
        prompt, _ = self.prompter.one_shot_prompt_with_thread(message, self.chat_frame)
        response, success = super().chat_response(prompt, max_tokens, stream)
        if success:
            content = get_str_from_response(response, self.model)
            self.add_message_to_thread(role="assistant", content=content)
        return response, success

    def get_conversation_history(self) -> pl.DataFrame:
        return self.chat_frame.memory_thread

    def get_last_user_message(self) -> pl.DataFrame:
        return self.chat_frame.last_message(role="user")

    def get_last_assistant_message(self) -> pl.DataFrame:
        return self.chat_frame.last_message(role="assistant")
