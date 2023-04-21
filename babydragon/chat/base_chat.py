import gradio as gr
import openai
import tiktoken 
from IPython.display import display, Markdown
from babydragon.utils.oai import mark_question, mark_system, get_mark_from_response , get_str_from_response
from babydragon.chat.prompts.default_prompts import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT
from typing import Callable, Tuple, List
from typing import List, Tuple, Dict


class Prompter:
    """
    This class handles the system and user prompts and the prompt_func. By subclassing and overriding the
    prompt_func, you can change the way the prompts are composed.
    """

    def __init__(self, system_prompt: str = None, user_prompt: str = None):
        """
        Initialize the Prompter with system and user prompts.

        :param system_prompt: A string representing the system prompt.
        :param user_prompt: A string representing the user prompt.
        """
        if system_prompt is None:
            self.system_prompt = DEFAULT_SYSTEM_PROMPT
            self.user_defined_system_prompt = False
        else:
            self.system_prompt = system_prompt
            self.user_defined_system_prompt = True
        if user_prompt is None:
            self.user_prompt = self.default_user_prompt
            self.user_defined_user_prompt = False
        else:
            self.user_prompt = user_prompt
            self.user_defined_user_prompt = True
            
        self.prompt_func: Callable[[str], Tuple[List[str], str]] = self.one_shot_prompt

    def default_user_prompt(self, message: str) -> str:
        return DEFAULT_USER_PROMPT.format(question=message)

    def one_shot_prompt(self, message: str) -> Tuple[List[str], str]:
        """
        Compose the prompt for the chat-gpt API.

        :param message: A string representing the user message.
        :return: A tuple containing a list of strings representing the prompt and a string representing the marked question.
        """
        marked_question = mark_question(self.user_prompt(message))
        prompt = [mark_system(self.system_prompt)] + [marked_question]
        return prompt, marked_question
    
    def update_system_prompt(self, new_prompt: str) -> None:
        """
        Update the system prompt.

        :param new_prompt: A string representing the new system prompt.
        """
        self.system_prompt = new_prompt

    def update_user_prompt(self, new_prompt: str) -> None:
        """
        Update the user prompt.

        :param new_prompt: A string representing the new user prompt.
        """
        self.user_prompt = new_prompt

class BaseChat:
    """
    This is the base class for chatbots, defining the basic functions that a chatbot should have, mainly the calls to
    chat-gpt API, and a basic Gradio interface. It has a prompt_func that acts as a placeholder for a call to chat-gpt
    API without any additional messages. It can be overridden by subclasses to add additional messages to the prompt.
    """
    def __init__(self, model: str = None, max_output_tokens: int = 1000):
        """
        Initialize the BaseChat with a model and max_output_tokens.

        :param model: A string representing the chat model to be used.
        :param max_output_tokens: An integer representing the maximum number of output tokens.
        """
        if model is None:
            self.model = "gpt-3.5-turbo"
        else:
            self.model = model
        self.tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')
        self.max_output_tokens = max_output_tokens
        self.failed_responses = []
        self.outputs = []
        self.inputs = []
        self.prompts = []
        self.prompt_func = self.identity_prompter

    def identity_prompter(self, message: str) -> Tuple[List[Dict], str]:
        """
        A simple identity prompter that takes a message and returns the message marked as a question.

        :param message: A string representing the user message.
        :return: A tuple containing the marked question and the original message.
        """
        return [mark_question(message)], mark_question(message)

    def chat_response(self, prompt: List[dict], max_tokens: int = None) -> Tuple[Dict, bool]:
        """
        Call the OpenAI API with the given prompt and maximum number of output tokens.

        :param prompt: A list of strings representing the prompt to send to the API.
        :param max_output_tokens: An integer representing the maximum number of output tokens.
        :return: A tuple containing the API response as a dictionary and a boolean indicating success.
        """
        if max_tokens is None:
            max_tokens = self.max_output_tokens
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=prompt,
                max_tokens=max_tokens,
            )
            return response, True
        
        except openai.error.APIError as e:
            print(e)
            fail_response = {"choices": [{"message": {"content": "I am sorry, I am having trouble understanding you. There might be an alien invasion interfering with my communicaiton with OpenAI."}}]}
            self.failed_responses.append(fail_response)
            return fail_response , False

    def reply(self, message: str, verbose : bool = True) -> str:
        """
        Reply to a given message using the chatbot.

        :param message: A string representing the user message.
        :return: A string representing the chatbot's response.
        """
        return self.query(message, verbose)["content"]
    
    def query(self, message: str, verbose: bool = True) -> str:
        """
        Query the chatbot with a given message, optionally showing the input and output messages as Markdown.

        :param message: A string representing the user message.
        :param verbose: A boolean indicating whether to display input and output messages as Markdown.
        :return: A string representing the chatbot's response.
        """
        
        prompt, _ = self.prompt_func(message)
        response, success = self.chat_response(prompt)
        if verbose:
            display(Markdown("#### Question: \n {question}".format(question = message)))
        if success:
            answer = get_mark_from_response(response)
            self.outputs.append(answer)
            self.inputs.append(message)
            self.prompts.append(prompt)
            if verbose:
                display(Markdown(" #### Anwser: \n {answer}".format(answer = get_str_from_response(response)))) 
            return answer
        else:
            raise Exception("OpenAI API Error inside query function")

    def run_text(self, text: str, state: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        Process the user's text input and update the chat state.

        :param text: A string representing the user input.
        :param state: A list of tuples representing the current chat state.
        :return: A tuple containing the updated chat state as two lists of tuples.
        """
        print("===============Running run_text =============")
        print("Inputs:", text)
        try: 
            print("======>Current memory:\n %s" % self.memory_thread)
        except:
            print("======>No memory")    
        response = self.reply(text)    
        state = state + [(text, response)]
        print("Outputs:", state)
        return state, state

    def gradio(self):
        """
        Create and launch a Gradio interface for the chatbot.
        """
        with gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}") as demo:
            chatbot = gr.Chatbot(elem_id="chatbot", label="NeuralDragonAI Alpha-V0.1")
            state = gr.State([])
            with gr.Row():
                with gr.Column(scale=1):
                    txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter, or upload an image").style(container=False)
                with gr.Column(scale=0.15, min_width=0):
                    clear = gr.Button("Clear️")

            txt.submit(self.run_text, [txt, state], [chatbot, state])
            txt.submit(lambda: "", None, txt)        
            demo.launch(server_name="localhost", server_port=7860 )