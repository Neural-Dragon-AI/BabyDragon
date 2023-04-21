from babydragon.memory.threads.fifo_thread import FifoThread
from babydragon.memory.threads.vector_thread import VectorThread
from babydragon.chat.chat import BaseChat, Prompter
from babydragon.oai_utils.utils import mark_question, mark_system, mark_answer



class FifoChat(FifoThread, BaseChat, Prompter):
    """FIFO Memory Thread, the oldest messages are removed first when reaching the max_memory limit, the memory is defined in terms of tokens,
    outs are passed to the longterm_memory, the lucid_memory is a redundant memory that stores all the messages"""
    def __init__(self, system_prompt=None, user_prompt= None, name='fifo_memory', max_fifo_memory=2048, max_output_tokens=1000, longterm_thread=None):
        FifoThread.__init__(self, name=name, max_memory=max_fifo_memory, longterm_thread=longterm_thread)
        BaseChat.__init__(self, max_output_tokens=max_output_tokens)
        Prompter.__init__(self, system_prompt=system_prompt, user_prompt=user_prompt)
        self.prompt_func = self.fifo_memory_prompt

    def fifo_memory_prompt(self, message):
        # Compose the prompt for the chat-gpt api
        prompt = [mark_system(self.system_prompt)] + self.memory_thread + [mark_question(self.user_prompt.format(question=message))]
        return prompt, mark_question(self.user_prompt.format(question=message))
    
    def query(self, question, verbose=True):
        """ Query the chatbot, the question is added to the memory and the answer is returned and added to the memory"""
        # First call the base class's query method
        answer = BaseChat.query(self, message=question, verbose=verbose)
        marked_question = mark_question(question)
        # Add the marked question and answer to the memory
        self.add_message(marked_question)
        self.add_message(answer)

        return answer

class VectorChat(VectorThread, BaseChat, Prompter):
    """ Vector Memory combined with chat memory_prompt is constructed by filling the memory with the k most similar messages to the question till the max prompt memory tokens are reached"""
    def __init__(self, index=None, name='vector_memory', max_vector_memory=2048, max_output_tokens=1000, system_prompt=None, user_prompt=None):
        VectorThread.__init__(self, index=index, name=name, max_context=max_vector_memory)
        BaseChat.__init__(self, max_output_tokens=max_output_tokens)
        Prompter.__init__(self, system_prompt=system_prompt, user_prompt=user_prompt)
        self.max_vector_memory = self.max_context
        self.prompt_func = self.vector_memory_prompt

    def vector_memory_prompt(self, question, k=10,):
        "combines system prompt, k most similar messages to the question and the user prompt"
        sorted_messages, sorted_scores, sorted_indices = self.sorted_query(question, k=k, max_tokens = self.max_vector_memory,reverse_order=True) 
        prompt = [mark_system(self.system_prompt)] + sorted_messages + [mark_question(self.user_prompt.format(question=question))]
        return prompt, mark_question(self.user_prompt.format(question=question))
    
    def weighted_memory_prompt(self, question, k=10, decay_factor = 0.1, temporal_weight = 0.5):
        weighted_messages, weighted_scores, weighted_indices = self.weighted_query(question, k=k, max_tokens = self.max_vector_memory,decay_factor = decay_factor, temporal_weight = temporal_weight, order_by = 'chronological', reverse = True)
        prompt = [mark_system(self.system_prompt)] + weighted_messages + [mark_question(self.user_prompt.format(question=question))]
        return prompt, mark_question(self.user_prompt.format(question=question))

    def query(self, question, verbose=False):
        """ Query the chatbot, the question is added to the memory and the answer is returned and added to the memory"""
        # First call the base class's query method
        answer = BaseChat.query(self, message=question, verbose=verbose)
        marked_question = mark_question(question)
        # Add the marked question and answer to the memory
        self.add_message(marked_question)
        self.add_message(answer)
        return answer


class FifoVectorChat(FifoThread, BaseChat, Prompter):
    def __init__(self, system_prompt=None, user_prompt=None, name='fifo_vector_memory', max_memory=2048, max_output_tokens= 1000, longterm_thread=None, longterm_frac=0.5):
        self.total_max_memory = max_memory

        self.setup_longterm_memory(longterm_thread, max_memory, longterm_frac)
        FifoThread.__init__(self, name=name, max_memory=self.max_short_term_memory, longterm_thread=self.longterm_thread)
        BaseChat.__init__(self, max_output_tokens=max_output_tokens)
        Prompter.__init__(self, system_prompt=system_prompt, user_prompt=user_prompt)
        self.prompt_func = self.fifovector_memory_prompt
        self.prompt_list = []

    def setup_longterm_memory(self, longterm_thread, max_memory, longterm_frac):
        if longterm_thread is None:
            self.longterm_frac = longterm_frac
            self.max_fifo_memory = int(max_memory * (1-self.longterm_frac))
            self.max_vector_memory = max_memory - self.max_fifo_memory    
            self.longterm_thread = VectorThread(None, 'longterm_memory', max_context=self.max_vector_memory)
        else:
            self.longterm_thread = longterm_thread
            self.max_vector_memory = self.longterm_thread.max_context
            self.max_fifo_memory = self.total_max_memory - self.max_longterm_memory
            self.longterm_frac = self.max_vector_memory / self.total_max_memory

    def fifovector_memory_prompt(self, question, k=10):
        prompt = [mark_system(self.system_prompt)]
        if len(self.longterm_thread.memory_thread) > 0 and self.longterm_thread.total_tokens <= self.max_vector_memory:
            prompt += self.longterm_thread.memory_thread
        elif len(self.longterm_thread.memory_thread) > 0 and self.longterm_thread.total_tokens > self.max_vector_memory:
            sorted_messages, sorted_scores, sorted_indices = self.longterm_thread.sorted_query(question, k=k, max_tokens = self.max_vector_memory,reverse_order=True) 
            prompt += sorted_messages

        prompt += self.memory_thread
        prompt += [mark_question(self.user_prompt.format(question=question))]
        return prompt, mark_question(self.user_prompt.format(question=question))

    def query(self, question, verbose=False):
        answer = BaseChat.query(self, message=question, verbose=verbose)
        marked_question = mark_question(question)
        self.add_message(marked_question)
        self.add_message(answer)
        return answer
    



        
