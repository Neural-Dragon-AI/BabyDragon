import gradio as gr
import openai
import tiktoken 
from IPython.display import display, Markdown
from babydragon.oai_utils.utils import mark_question, mark_system, mark_answer

class Chat:
    """this is the base class for chatbots, it defines the basic functions that a chatbot should have, mainly the calls to chat-gpt api, and a basic gradio interface, you need to create a sub-class to connect it to a memory thread"""
    def __init__(self,system_prompt:str = None, user_prompt:str = None, max_output_tokens = 1000):
        self.model = "gpt-3.5-turbo"
        self.tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')
        self.max_output_tokens = max_output_tokens
        if system_prompt is None:
            self.system_prompt = self.get_default_system_prompt()
        if  user_prompt is None:
            self.user_prompt = self.get_default_user_prompt()
        self.failed_responses = []
        self.prompt_func = self.one_shot_prompt
        self.answers = []

    def get_mark_from_response(self, response):
        #return the answer from the response
        role = response['choices'][0]["message"]["role"]
        message = response['choices'][0]["message"]["content"]
        return {"role": role, "content": message}
    def get_str_from_response(self, response):
        #return the answer from the response
        return response['choices'][0]["message"]["content"]
        
    def get_default_system_prompt(self):
        one_shot_prompt= "You are a useful Assistant you role is to answer questions in an exhaustive way! Please be helpful to the user he loves you!"
        return one_shot_prompt
    
    def get_default_user_prompt(self):
        empty_user_prompt = "{question}"
        return empty_user_prompt 
    
    def one_shot_prompt(self, message):
        #compose the prompt for the chat-gpt api
        prompt = [mark_system(self.system_prompt)]+ [mark_question(self.user_prompt.format(question=message))]
        return prompt, mark_question(self.user_prompt.format(question=message))

    def chat_response(self,prompt):
        if type(prompt) is str:
            prompt, _ = self.prompt_func(prompt)
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=prompt,
                max_tokens=self.max_output_tokens,
            )
            return response, True
        except openai.error.APIError as e:
            print(e)
            fail_response = {"choices": [{"message": {"content": "I am sorry, I am having trouble understanding you. There might be an alien invasion interfering with my communicaiton with OpenAI."}}]}
            self.failed_responses.append(fail_response)
            return fail_response , False

    def query(self, message):
        """ overwritten by sub-classes to add memory to the chatbot"""
        prompt, _ = self.prompt_func(message)
        response, success = self.chat_response(prompt)
        display(Markdown("#### Question: \n {question}".format(question = message)))
        if success:
            self.answers.append(self.get_mark_from_response(response))
            display(Markdown(" #### Anwser: \n {answer}".format(answer = self.get_str_from_response(response)))) 
            return self.answers[-1]

    def reply(self,question):
        #wrapprer for query that only returns the answer as a string
        return self.query(question)["content"]    

    def run_text(self, text, state):
        print("===============Running run_text =============")
        print("Inputs:", text)
        try: 
            print("======>Current memory:\n %s" % self.memory_thread)
        except:
            print("======>No memory")    
        answer = self.query(text)    
        response = answer["content"]
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
            demo.launch(server_name="localhost", server_port=7860  )          