from babydragon.core.memory_thread import FifoMemory, VectorMemory
from babydragon.chat.chat import Chat
from babydragon.oai_utils.utils import mark_question, mark_system, mark_answer

class FifoChat(FifoMemory, Chat):
    """FIFO Memory Thread, the oldest messages are removed first when reaching the max_memory limit, the memory is defined in terms of tokens,
    outs are passed to the longterm_memory, the lucid_memory is a redundant memory that stores all the messages"""

    def __init__(self, system_prompt= None , name= 'fifo_memory', max_memory = 2048, longterm_thread = None):
        super().__init__(name, max_memory, longterm_thread)
        if system_prompt is None:
            self.system_prompt = self.get_default_system_prompt()
        else:
            self.system_prompt = system_prompt
        Chat.__init__(self, self.system_prompt)
        self.prompt_func = self.fifo_memory_prompt

    def fifo_memory_prompt(self, message):
        #compose the prompt for the chat-gpt api
        prompt = [mark_system(self.system_prompt)]+ self.memory_thread + [mark_question(self.user_prompt.format(question=message))]
        
        return prompt, mark_question(self.user_prompt.format(question=message))
        
    def query(self, question):
        #compose the prompt for the chat-gpt api
        prompt, marked_question = self.prompt_func(question)
        #call the chat-gpt api
        response, success = self.chat_response(prompt)
        if success:
            #add the question and answer to the chat_history
            answer = self.get_mark_from_response(response)
            self.add_message(marked_question)
            self.add_message(answer)
            #get the answer from the open ai response
            return answer
        else:
            return response.choices[0].message.content
        

class VectorChat(VectorMemory, Chat):
    """ Vector Memory combined with chat memory_prompt is constructed by filling the memory with the k most similar messages to the question till the max prompt memory tokens are reached"""
    def __init__(self, index=None, name='vector_memory', max_context = 2048, system_prompt = None, user_prompt = None):
        super().__init__(index, name, max_context)
        if system_prompt is None:
            self.system_prompt = self.get_default_system_prompt()
        Chat.__init__(self, self.system_prompt, user_prompt)
        self.prompt_func = self.vector_memory_prompt

    def vector_memory_prompt(self, question, k = 10):
        #starts by retieving the k most similar messages to the question
        # then starting from the most similar message it adds the messages to the prompt till the max_prompt is reached
        # the prompt is composed by the system prompt, the messages in the memory and the question
        # the marked question is the last message in the prompt
        
        prompt = [mark_system(self.system_prompt)]
        prompt +=  self.get_token_bound_prompt(question, k = k)
        prompt+=[mark_question(self.user_prompt.format(question=question))]
        
        return prompt, mark_question(self.user_prompt.format(question=question))
    
    def query(self, question,verbose = False):
        #compose the prompt for the chat-gpt api
        prompt, marked_question = self.prompt_func(question)

        if verbose:
            print("prompt: ", prompt)
        #call the chat-gpt api
        response, success = self.chat_response(prompt)
        if success:
            #add the question and answer to the chat_history
            answer = self.get_mark_from_response(response)
            self.add_message(marked_question)
            self.add_message(answer)
            #get the answer from the open ai response
            return answer
        else:
            return response.choices[0].message.content
        
class FifoVectorChat(FifoMemory,Chat):
    def __init__(self, system_prompt= None , name= 'fifo_vector_memory', max_memory = 2048, longterm_thread = None, longterm_frac = 0.5):
        self.total_max_memory = max_memory
        self.setup_longterm_memory(longterm_thread, max_memory , longterm_frac)
        
        super().__init__(name, self.max_short_term_memory, self.longterm_thread)
        if system_prompt is None:
            self.system_prompt = self.get_default_system_prompt()
        Chat.__init__(self, self.system_prompt)
        self.prompt_func = self.fifovector_memory_prompt
        self.prompt_list = []

    def setup_longterm_memory(self, longterm_thread, max_memory , longterm_frac):
        if longterm_thread is None:
            self.longterm_frac = longterm_frac
            self.max_short_term_memory =int(max_memory * (1-self.longterm_frac))
            self.max_longterm_memory = max_memory - self.max_short_term_memory    
            self.longterm_thread = VectorMemory(None, 'longterm_memory',max_context = self.max_longterm_memory)
        else:
            self.longterm_thread = longterm_thread
            self.max_longterm_memory = self.longterm_thread.max_context
            self.max_short_term_memory = self.total_max_memory - self.max_longterm_memory
            self.longterm_frac = self.max_longterm_memory/self.total_max_memory
    
    def fifovector_memory_prompt(self, question, k = 10):
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
        prompt+=[mark_question(self.user_prompt.format(question=question))]
        return prompt, mark_question(self.user_prompt.format(question=question))

    def query(self, question):
        #compose the prompt for the chat-gpt api
        prompt, marked_question = self.prompt_func(question)
        self.prompt_list.append(prompt)
        #call the chat-gpt api
        response, success = self.chat_response(prompt)
        if success:
            #add the question and answer to the chat_history
            answer = self.get_mark_from_response(response)
            self.add_message(marked_question)
            self.add_message(answer)
            #get the answer from the open ai response
            return answer
        else:
            return response.choices[0].message.content
    
