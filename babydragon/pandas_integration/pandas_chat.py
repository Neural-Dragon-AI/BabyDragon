from babydragon.chat.chat import Chat
from babydragon.pandas_integration.pandas_index import PandasIndex
from babydragon.chat.memory_chat import FifoVectorChat
import tiktoken
from babydragon.oai_utils.utils import mark_question, mark_system, mark_answer

class PandasChat(PandasIndex, Chat):
    """ combines a chat with a panda index such that the chat response are based on the content of the PandasIndex"""
    def __init__(self, pandaframe, max_context, max_output_tokens, index_description=None, 
                 columns=None, name='panda_index', save_path=None, in_place=True, embeddings_col=None):
        # Initialize PandasIndex
        PandasIndex.__init__(self, pandaframe, columns, name, save_path, in_place, embeddings_col)

        self.max_context = max_context

        # Initialize Chat
        self.tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')
        Chat.__init__(self, max_output_tokens=max_output_tokens)

        self.prompt_func = self.panda_prompt
        self.system_prompt = self.get_default_panda_prompt(index_description)

    def get_default_panda_prompt(self, index_description):
        system_prompt = """You are a Chatbot assistant that can use a external knowledge base to answer questions.
        The user will always add hints from the external knowledge base. 
        You express your thoughts using princpled reasoning and always pay attention to the
         hints.  Your knowledge base description is {index_descrpiton}:"""
        return system_prompt.format(index_descrpiton = index_description)
    
    def get_hint_prompt(self, question):
        hints = self.get_token_bound_hints(question, k = 10, max_context = self.max_context)
        hints_string = "\n ".join(hints)
        prefix= "I am going to ask you a question and you should use the hints to answer it. The hints are:\n{hints_string}"
        questionintro ="The question is: {question}"
        return prefix.format(hints_string = hints_string) + questionintro.format(question = question)
    
    def panda_prompt(self, question):
        #compose the prompt for the chat-gpt api
        # the prompt is composed by the system_prompt, the top-k most similar messages to the question and the question

        prompt = [mark_system(self.system_prompt)]
        
        prompt += [mark_question(self.get_hint_prompt(question))]
        #display prompt
        # display(Markdown(str(prompt)))
        return prompt, mark_question(question)

    
    
# here we write a fifo vectorchat with a PandasIndex as external source of information, we can not subclass PandasIndex because many 
# methods are overlapping 

class FifoVectorPandasChat(FifoVectorChat):

    def __init__(self,pandaframe,columns,embeddings_col = None, system_prompt=None, name='fifovec_panda_memory',max_context=4000, max_memory=2048, longterm_thread=None, longterm_frac=0.5):
        super().__init__(system_prompt, name, max_memory, longterm_thread, longterm_frac)
        self.PandasIndex = PandasChat(pandaframe,columns = columns, max_context = max_context, max_output_tokens = 100, index_description = "alice_pandraframe", embeddings_col = embeddings_col)
        self.prompt_func = self.memory_panda_prompt
        self.max_output_tokens = 100
        self.model = "gpt-4"
    


    def memory_panda_prompt(self, question, k = 10):
        # compose the prompt for the chat-gpt api
        # the first half of the prompt is composed by long term memory with up to max_longterm_memory tokens
        # the second half ot the prompt is composed by the fifo memory with up to max_short_term_memory tokens
        # the prompt is composed by the system prompt, the messages in the memory and the question

        prompt = [mark_system(self.system_prompt)]
        #check if something is in the long term memory and if it is smaller than the max_longterm_memory
        if len(self.longterm_thread.memory_thread) > 0 and self.longterm_thread.total_tokens <= self.max_longterm_memory:
            #add all the messages in the long term memory
            prompt+=self.longterm_thread.memory_thread
        elif len(self.longterm_thread.memory_thread) > 0 and self.longterm_thread.total_tokens > self.max_longterm_memory:
            # if the long term memory is bigger than the max_longterm_memory then add the k most similar messages to the question till the max_longterm_memory is reached
            prompt += self.longterm_thread.get_token_bound_prompt(question, k =k)
        
        # add the complete short term memory to the prompt because it is a fifo memory is always smaller than the max_short_term_memory
        prompt+=self.memory_thread
        prompt += [mark_question(self.PandasIndex.get_hint_prompt(question))]
        return prompt, mark_question(question)
    