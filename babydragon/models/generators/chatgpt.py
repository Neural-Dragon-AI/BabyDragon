import openai
from typing import List, Tuple, Dict

def chatgpt_response(prompt: List[dict],model: str = "gpt-3.5-turbo", max_tokens: int = 1000
    ) -> Tuple[Dict, bool]:
        """
        Call the OpenAI API with the given prompt and maximum number of output tokens.

        :param prompt: A list of strings representing the prompt to send to the API.
        :param max_output_tokens: An integer representing the maximum number of output tokens.
        :return: A tuple containing the API response as a dictionary and a boolean indicating success.
        """             
        try:
            print("Trying to call OpenAI API...")
            response = openai.ChatCompletion.create(
                model=model,
                messages=prompt,
                max_tokens=max_tokens,
            )
            return response, True

        except openai.error.APIError as e:
            print(e)
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