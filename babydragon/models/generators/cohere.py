import cohere
co = cohere.Client('LeohkffIg5ucAxbMSfiCIhZ0RL9M2uuw0GVb99ZN')
from typing import List, Tuple, Dict
from babydragon.utils.chatml import convert_mark_to_str_prompt



def cohere_response(prompt: List[dict],model: str = "command", max_tokens: int = 1000
    ) -> Tuple[Dict, bool]:
        """
        Call the Cohere API with the given prompt and maximum number of output tokens.
        

        :param prompt: A list of strings representing the prompt to send to the API.
        :param max_output_tokens: An integer representing the maximum number of output tokens.
        :param model: A string representing the model to use. either command or command-nightly
        :return: A tuple containing the API response cohere object and a boolean indicating success.
        """             
        try:
            prompt= convert_mark_to_str_prompt(prompt)
            print("Trying to call Cohere API... using model:", model)
            response = co.generate(
            model= model,
            prompt= prompt,
            max_tokens=max_tokens,
            #end_sequences=['#SYSTEM:', '#USER:'],
            )
            return response, True

        except:
              return None, False
        

def cohere_summarize(prompt: str, model: str = "summarize-xlarge", length: str = "medium", extractiveness: str = "medium", format: str = "bullets", additional_command: str = None) -> str:
    response = co.summarize( 
    text=prompt,model=model, 
    length=length,
    extractiveness=extractiveness,
    format=format,
    additional_command=additional_command
    )

    summary = response.summary
    return summary
              
