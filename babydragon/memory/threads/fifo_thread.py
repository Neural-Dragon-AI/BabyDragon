from IPython.display import display, Markdown
from babydragon.memory.threads.base_thread import BaseThread
from babydragon.utils.oai import check_dict
import copy

class FifoThread(BaseThread):
    """FIFO Memory BaseThread, the oldest messages are removed first when reaching the max_memory limit, the memory is defined in terms of tokens, 
    outs are passe to the longterm_memory, 
    lucid_memory is a redundant memory that stores all the messages
    """
    def __init__(self, name= 'fifo_memory', max_memory = None, longterm_thread = None, redundant = True):
        
        BaseThread.__init__(self,name= name , max_memory= None)
        if redundant is True:            
            self.redundant_thread = BaseThread(name = 'lucid_memory',max_memory = None)
        else:
            self.redundant_thread = None
        if longterm_thread is None:
            self.longterm_thread = BaseThread(name ='longterm_memory',max_memory = None)
        else:
            self.longterm_thread = longterm_thread
        # create an alias for the memory_thread to make the code more readable
        self.fifo_thread = self.memory_thread
            
    def to_longterm(self, idx: int):
        """ move the message at the index idx to the longterm_memory"""
        #move the message at the index idx to the longterm_memory
        display(Markdown("The memory BaseThread is full, the message with index {} was moved to the longterm memory".format(idx)))
        message = copy.deepcopy(self.memory_thread[idx])
        # print("preso il messagio e provo a ad aggiungerlo al longterm", message)
        self.longterm_thread.add_message(message)
        self.remove_message(idx=idx)
    
    def add_message(self, message_dict: dict):
        """ add a message to the memory_thread, if the memory_thread is full remove the oldest message from the memory_thread using the FIFO principle, if not enough space is available remove the oldest messages until enough space is available"""
        # message_dict = {"role": role, "content": content}
        #chek that the message_dict is a dictionary or a list of dictionaries
        message_dict = check_dict(message_dict)
        if self.redundant_thread is not None:
            self.redundant_thread.add_message(message_dict)
        message_tokens = self.get_message_tokens(message_dict)
        
        if self.total_tokens + message_tokens > self.max_memory:
            while self.total_tokens + message_tokens > self.max_memory and len(self.memory_thread) > 0:
                
                self.to_longterm(idx=0)
            super().add_message(message_dict)
            
        else:
            #add the message_dict to the memory_thread
            # update the total number of tokens
            super().add_message(message_dict)