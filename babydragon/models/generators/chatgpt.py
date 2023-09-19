import logging
from typing import Dict, Generator, List, Tuple, Union

import openai

logging.basicConfig(level=logging.INFO)


def chatgpt_response(
    prompt: List[Dict[str, Union[str, Dict[str, str]]]],
    model: str = "gpt-3.5-turbo",
    max_tokens: int = 1000,
    stream: bool = False,
) -> Union[Generator, Tuple]:
    try:
        print("Trying to call OpenAI API...")
        response = openai.ChatCompletion.create(
            model=model, messages=prompt, max_tokens=max_tokens, stream=stream
        )
        return response, True

    except openai.APIError as e:
        logging.error(f"Unexpected error in openai call: {e}")
        fail_response = {
            "choices": [
                {
                    "message": {
                        "content": "I am sorry, I am having trouble understanding you. There might be an alien invasion interfering with my communicaiton with OpenAI."
                    }
                }
            ]
        }
        return fail_response, False
