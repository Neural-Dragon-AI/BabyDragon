import gradio as gr
import openai
import tiktoken 
from IPython.display import display, Markdown
from babydragon.oai_utils.utils import mark_question, mark_system, mark_answer, get_mark_from_response , get_str_from_response, check_dict
from babydragon.chat.prompts.default_prompts import default_system_prompt, default_user_prompt


class Prompter:
    """This class handles the system and user prompts and the prompt_func, by subclassing and overriding the prompt_func you can change the way the prompts are composed"""

    def __init__(self, system_prompt=None, user_prompt=None):
        if system_prompt is None:
            self.system_prompt = default_system_prompt
        if user_prompt is None:
            self.user_prompt = default_user_prompt
        self.prompt_func = self.one_shot_prompt

    def one_shot_prompt(self, message):
        #compose the prompt for the chat-gpt api
        prompt = [mark_system(self.system_prompt)]+ [mark_question(self.user_prompt.format(question=message))]
        return prompt, mark_question(self.user_prompt.format(question=message))

    def update_system_prompt(self, new_prompt):
        self.system_prompt = new_prompt

    def update_user_prompt(self, new_prompt):
        self.user_prompt = new_prompt


class BaseChat:
    """this is the base class for chatbots, it defines the basic functions that a chatbot should have, mainly the calls to chat-gpt api, and a basic gradio interface
    it has a prompt func that acts as a placeholder for a call to chat-gpt api without any additional messages, it can be overriden by subclasses to add additional messages to the prompt"""
    def __init__(self, model, max_output_tokens = 1000):
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

    def identity_prompter(self, message):
        return mark_question(message), message

    def chat_response(self, prompt, max_output_tokens = None):
        if max_tokens is None:
            max_tokens = self.max_output_tokens
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=prompt,
                max_tokens=max_output_tokens,
            )
            return response, True
        
        except openai.error.APIError as e:
            print(e)
            fail_response = {"choices": [{"message": {"content": "I am sorry, I am having trouble understanding you. There might be an alien invasion interfering with my communicaiton with OpenAI."}}]}
            self.failed_responses.append(fail_response)
            return fail_response , False

    def reply(self,message):
        return self.query(message)["content"]
    
    def query(self, message, verbose = True):
        self.inputs.append(message)
        prompt, _ = self.prompt_func(message)
        self.prompts.append(str(prompt))
        response, success = self.chat_response(prompt)
        if verbose:
            display(Markdown("#### Question: \n {question}".format(question = message)))
        if success:
            answer = get_mark_from_response(response)
            self.outputs.append(answer)
            if verbose:
                display(Markdown(" #### Anwser: \n {answer}".format(answer = self.get_str_from_response(response)))) 
            return answer
        else:
            raise Exception("OpenAI API Error inside query function")

    def run_text(self, text, state):
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
    

class Chat(BaseChat, Prompter):
    """This class combines the BaseChat and Prompter classes to create a oneshot chatbot with a system and user prompt"""
    def __init__(self, max_output_tokens=1000, system_prompt=None, user_prompt=None):
        BaseChat.__init__(self, max_output_tokens=max_output_tokens)
        Prompter.__init__(self, system_prompt=system_prompt, user_prompt=user_prompt)



