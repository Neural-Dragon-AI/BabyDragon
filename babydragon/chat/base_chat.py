from typing import Callable, Dict, List, Tuple, Union, Generator
import gradio as gr
import openai
import tiktoken
from IPython.display import Markdown, display
from babydragon.models.generators.chatgpt import chatgpt_response
from babydragon.models.generators.cohere import cohere_response
from babydragon.chat.prompts.default_prompts import (DEFAULT_SYSTEM_PROMPT,
                                                     DEFAULT_USER_PROMPT)
from babydragon.utils.chatml import (get_mark_from_response,
                                  get_str_from_response, mark_question,
                                  mark_system)
import logging


class Prompter:
    """
    This class handles the system and user prompts and the prompt_func. By subclassing and overriding the
    prompt_func, you can change the way the prompts are composed.
    """

    def __init__(self, system_prompt: Union[str,None] = None, user_prompt: Union[str,None] = None):
        """
        Initialize the Prompter with system and user prompts.

        :param system_prompt: A string representing the system prompt.
        :param user_prompt: A string representing the user prompt.
        """
        
        if system_prompt is None:
            self.system_prompt = DEFAULT_SYSTEM_PROMPT
            self.user_defined_system_prompt = None
        else:
            self.system_prompt = system_prompt
            self.user_defined_system_prompt = system_prompt
        if user_prompt is None:
            self.user_prompt = self.default_user_prompt
            self.user_defined_user_prompt = None
        else:
            self.user_prompt = user_prompt
            self.user_defined_user_prompt = user_prompt

        self.prompt_func: Callable[[str], Tuple[List[str], str]] = self.one_shot_prompt
        self.user_defined_ids = []
        self.user_defined_values = []
        self.use_user_defined_ids = False

    def set_default_prompts(self):
        self.system_prompt = DEFAULT_SYSTEM_PROMPT
        self.user_prompt = self.default_user_prompt


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

    def update_user_prompt(self, new_prompt ) -> None:
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

    def __init__(self, model: Union[str,None] = None, max_output_tokens: int = 200):
        """
        Initialize the BaseChat with a model and max_output_tokens.

        :param model: A string representing the chat model to be used.
        :param max_output_tokens: An integer representing the maximum number of output tokens.
        """
        if model is None:
            self.model = "gpt-3.5-turbo"
        else:
            self.model = model
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.max_output_tokens = max_output_tokens
        self.failed_responses = []
        self.outputs = []
        self.inputs = []
        self.prompts = []
        self.prompt_func = self.identity_prompter

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the tokenizer attribute from the state
        del state["tokenizer"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reinitialize the tokenizer attribute after unpickling
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def identity_prompter(self, message: str) -> Tuple[List[Dict], str]:
        """
        A simple identity prompter that takes a message and returns the message marked as a question.

        :param message: A string representing the user message.
        :return: A tuple containing the marked question and the original message.
        """
        return [mark_question(message)], mark_question(message)

    def chat_response( self, prompt: List[dict], max_tokens: Union[int,None] = None, stream:bool = False ) -> Union[Generator,Tuple[Dict, bool]]:
        if max_tokens is None:
            max_tokens = self.max_output_tokens
        if "gpt" in self.model:
            logging.info(prompt)
            response, status = chatgpt_response(prompt=prompt,model=self.model, max_tokens = 1000, stream=stream)
            if status:
                return response, True
            else:
                self.failed_responses.append(response)
                return response, False

        elif "command" in self.model:
            response, status = cohere_response(prompt=prompt,model=self.model, max_tokens = 1000)  
            if status:
                return response, True
            else:
                self.failed_responses.append(response)
                return response, False
        else:
            return {}, False 

    def reply(self, message: str, verbose: bool = True, stream: bool = False) -> Union[Generator, str]:
        """
        Reply to a given message using the chatbot.

        :param message: A string representing the user message.
        :return: A string representing the chatbot's response.
        """
        if stream:
            return self.query(message, verbose, stream)
        else:
            return self.query(message, verbose)["content"]

    def query(self, message: str, verbose: bool = True, stream: bool = False) -> Union[Generator, str]:
        """
        Query the chatbot with a given message, optionally showing the input and output messages as Markdown.

        :param message: A string representing the user message.
        :param verbose: A boolean indicating whether to display input and output messages as Markdown.
        :return: A string representing the chatbot's response.
        """

        prompt, _ = self.prompt_func(message)

        if stream:
            return self.chat_response(prompt=prompt, stream=stream)

        response, success = self.chat_response(prompt)
        if verbose:
            display(Markdown("#### Question: \n {question}".format(question=message)))
        if success:
            answer = get_mark_from_response(response, self.model)
            self.outputs.append(answer)
            self.inputs.append(message)
            self.prompts.append(prompt)
            if verbose:
                display(
                    Markdown(
                        " #### Anwser: \n {answer}".format(
                            answer=get_str_from_response(response, self.model)
                        )
                    )
                )
            return answer
        else:
            raise Exception("OpenAI API Error inside query function")

    def reset_logs(self):
        """
        Reset the chatbot's memory.
        """
        self.outputs = []
        self.inputs = []
        self.prompts = []

    def get_logs(self):
        """
        Get the chatbot's memory.

        :return: A tuple containing the chatbot's memory as three lists of strings.
        """
        return self.inputs, self.outputs, self.prompts

    def run_text(
        self, text: str, state: List[Tuple[str, str]]
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
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
        print("failed here")
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
                    txt = gr.Textbox(
                        show_label=False,
                        placeholder="Enter text and press enter, or upload an image",
                    ).style(container=False)
                with gr.Column(scale=0.15, min_width=0):
                    clear = gr.Button("Clear️")

            txt.submit(lambda text, state: self.run_text(text, state), [txt, state], [chatbot, state])
            txt.submit(lambda: "", None, txt)
            demo.launch(server_name="localhost", server_port=7860)
