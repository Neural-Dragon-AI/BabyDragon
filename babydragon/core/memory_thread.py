import tiktoken
from IPython.display import display, Markdown
import copy 
from babydragon.core.vector_index import MemoryIndex
from babydragon.oai_utils.utils import  check_dict, mark_question

class MemoryThread:
    """this class is used to keep track of the memory thread and the total number of tokens all memories should subclass this class
    if max_memory is None it has no limit to the number of tokens that can be stored in the memory thread """
    def __init__(self,name= 'memory',max_memory= None):
        self.name = name
        self.max_memory = max_memory
        self.memory_thread = []
        self.total_tokens = 0
        self.tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')
    def __getitem__(self, idx):
        return self.memory_thread[idx]    

        

    def get_message_tokens(self, message_dict):
        message = message_dict["content"]
        return len(self.tokenizer.encode(message))+6 # +6 for the role token

    def remove_message(self, message_dict=  None , idx = None):
        # if idx search in the memory_thread the latest message that matches the message_dict an
        #remove it from the memory_thread otherwise remove the message at the index idx
        # update the total number of tokens
        # return a boolean that indicates if the message was found and removed
        if message_dict is None and idx is None:
            raise Exception("You need to provide either a message_dict or an idx")
        
        if idx is None:
            message_dict = check_dict(message_dict)
            search_results = self.find_message(message_dict)
            if search_results is not None:
                idx = search_results[-1]["idx"]
                message = search_results[-1]["message"]
                self.memory_thread.pop(idx)
                self.total_tokens -= self.get_message_tokens(message)
            else:   
                raise Exception("The message was not found in the memory thread")
        else:
            if idx < len(self.memory_thread):
                message = self.memory_thread.pop(idx)
                self.total_tokens -= self.get_message_tokens(message)
            else:  
                raise Exception("The index was out bound")
    
    def add_message(self,message_dict: dict):
        # message_dict = {"role": role, "content": content}
        #chek that the message_dict is a dictionary or a list of dictionaries 
        message_tokens = self.get_message_tokens(message_dict)
        
        if  self.max_memory is None or self.total_tokens + message_tokens <= self.max_memory:
            #add the message_dict to the memory_thread
            # update the total number of tokens
            self.memory_thread.append(message_dict)
            self.total_tokens += message_tokens
            return True    
        else :
            display(Markdown("The memory thread is full, the last message was not added"))
            return False

    
                    
    def find_message(self,message_dict: dict, last=False):
        # search the memory_thread from start_idx to the end of the memory_thread for all the messages that match the message_dict
        # return a seach_results dictionary with the following structure [{"message": message, "idx": idx}]
        # if last is True return only the last message that matches the message_dict
        search_results = []
        message_dict = check_dict(message_dict)

        print("the message dict is ", message_dict, type(message_dict))
        for idx, message in enumerate(self.memory_thread):
            print("the index is ", idx, type(idx))
            print("the message is ", message, type(message))
            if message["role"] == message_dict["role"] and message["content"] == message_dict["content"]:
                search_results.append({"message": message, "idx": idx})
        if last and len(search_results) > 0:
            return [search_results[-1]]
        elif len(search_results) > 0:
            return search_results
        else:
            return None
    
    def length(self):
        #return the length of the memory_thread
        return len(self.memory_thread)
    
    def slice_tokens(self, start_idx= 0, end_idx = None ):
        #compute the tokens from start_idx to end_idx
        # default behavior is to compute the tokens of the whole memory_thread
        tokens = 0
        if end_idx is None:
            end_idx = len(self.memory_thread)
        try:
            for message in self.memory_thread[start_idx:end_idx]:
                tokens += self.get_message_tokens(message)
            return tokens
        except:
            ValueError ("The slice is not valid")
        
    def get_message(self, idx: int ):
        return self.memory_thread[idx]
    
    def get_thread(self):
        return self.memory_thread
    
    def slice(self,start,end):
        #return the memory_thread slice from start_idx to end_idx
        # default behavior is to return the whole memory_thread
        try:
            return self.memory_thread[start:end]
        except:
            ValueError ("The slice is not valid")

    def print(self):
        # detailed output of the memory_thread using markdown
        
        display(Markdown("## Memory Thread"))
        display(Markdown("#### Total Tokens: "+str(self.total_tokens)))
        display(Markdown("#### Max Tokens: "+str(self.max_memory)))
        display(Markdown("#### Number of Messages: "+str(len(self.memory_thread))))
        display(Markdown("#### Messages:"))
        for message in self.memory_thread:
            display(Markdown("#### "+message["role"]+": "+message["content"]))





class FifoMemory(MemoryThread):
    """FIFO Memory Thread, the oldest messages are removed first when reaching the max_memory limit, the memory is defined in terms of tokens, 
    outs are passe to the longterm_memory, 
    lucid_memory is a redundant memory that stores all the messages
    """
    def __init__(self, name= 'fifo_memory', max_memory = None, longterm_thread = None):
        
        super().__init__(name= name , max_memory= max_memory)
        self.lucid_thread = MemoryThread(name = 'lucid_memory',max_memory = None)
        if longterm_thread is None:
            self.longterm_thread = MemoryThread(name ='longterm_memory',max_memory = None)
        else:
            self.longterm_thread = longterm_thread
        # create an alias for the memory_thread to make the code more readable
        self.fifo_thread = self.memory_thread
        
        
    def to_longterm(self, idx):
        #move the message at the index idx to the longterm_memory
        display(Markdown("The memory thread is full, the oldest message with index {} was moved to the longterm memory".format(idx)))
        message = copy.deepcopy(self.memory_thread[idx])
        print("preso il messagio e provo a ad aggiungerlo al longterm", message)
        status = self.longterm_thread.add_message(message)
        if status:
            print("ho aggiunto il messaggio al longterm")
            self.remove_message(idx=idx)
        else:
            raise Exception("The longterm memory is bugged")    
        
    def add_message(self,message_dict: dict):
        # message_dict = {"role": role, "content": content}
        #chek that the message_dict is a dictionary or a list of dictionaries
        self.lucid_thread.add_message(message_dict)
        message_dict = check_dict(message_dict)
        message_tokens = self.get_message_tokens(message_dict)
        
        if self.total_tokens + message_tokens > self.max_memory:
            #remove the oldest message from the memory_thread using the FIFO principle, if not enough space is available remove the oldest messages using  until enough space is available
            while self.total_tokens + message_tokens > self.max_memory and len(self.memory_thread) > 0:
                #remove the oldest message from the memory_thread using the FIFO principle and add it to the longterm_memory
                
                self.to_longterm(idx=0)
            super().add_message(message_dict)
            return True
        else:
            #add the message_dict to the memory_thread
            # update the total number of tokens
            super().add_message(message_dict)
            return True 
        
class VectorMemory(MemoryThread, MemoryIndex):
    """ vector memory, creates a faiss index with the messages and allows to search for similar messages, memory threads can be composed in similarity order or in (TODO) chronological order 
    """
    def __init__(self, index = None, name= 'vector_memory', max_context = 2048):
        super().__init__(name= name , max_memory= None)
        MemoryIndex.__init__(self, index = index, name = name)
        self.max_context = max_context
        
    def index_message(self,message_dict: dict, verbose = True):
        """index a message in the faiss index, the message is embedded and the id is saved in the ids list
        """
        message_dict = check_dict(message_dict)
        self.add_to_index(value = message_dict, verbose = verbose)

    def add_message(self,message_dict: dict):
        print("checking the dict")
        message_dict = check_dict(message_dict)
        print("trying to add the message")
        super().add_message(message_dict)
        self.index_message(message_dict) 
        return True
    
    def get_token_bound_prompt(self, query, k = 10):
        prompt = []
        context_tokens = 0
        if len(self.memory_thread) > 0 and self.total_tokens > self.max_context:
            top_k = self.faiss_query(mark_question(query), k = len(self.memory_thread))
            # print("top_k: ", top_k)
            top_k_prompt = []
            for message in top_k:
                #mark the message and gets the length in tokens
                message_tokens = self.get_message_tokens(message)
                if context_tokens+message_tokens <= self.max_context:
                    top_k_prompt+=[message]
                    context_tokens += message_tokens
            #inver the top_k_prompt to start from the most similar message
            top_k_prompt.reverse()
            prompt+=top_k_prompt
            #reverse the prompt so that last is the most similar message
            prompt.reverse()
        elif len(self.memory_thread) > 0:
            prompt+=self.memory_thread    
        return prompt