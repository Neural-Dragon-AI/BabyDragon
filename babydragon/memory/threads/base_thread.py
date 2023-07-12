from time import time as now
from typing import Any, Optional, Union, Dict, List
import os
import tiktoken
import polars as pl
import requests
from bs4 import BeautifulSoup
import json
from babydragon.utils.chatml import check_dict


class BaseThread:
    """
    This class is used to keep track of the memory thread of a conversation and the total number of tokens.
    All conversation memories should subclass this class. If max_memory is None, it has
    no limit to the number of tokens that can be stored in the memory thread.
    """

    def __init__(
        self,
        name: str = "memory",
        max_memory: Optional[int] = None,
        tokenizer: Optional[Any] = None,
        save_path: str = 'threads'

    ) -> None:
        """
        Initialize the BaseThread instance.

        :param name: The name of the memory thread. Defaults to 'memory'.
        :param max_memory: The maximum number of tokens allowed in the memory thread.
                           Defaults to None, which means no limit.
        :param tokenizer: The tokenizer to be used for tokenizing messages.
                          Defaults to None, which means using the tiktoken encoding for the 'gpt-3.5-turbo' model.
        """
        self.name = name
        self.max_memory = max_memory
        self.memory_schema = {"role": pl.Utf8, "content": pl.Utf8,"timestamp":pl.Float64,"tokens_count":pl.UInt16}
        self.memory_thread: pl.DataFrame = pl.DataFrame(schema=self.memory_schema)
        """ self.time_stamps = [] """
        """ self.message_tokens = [] """
        self.total_tokens = self.get_total_tokens_from_thread()
        self.save_path = save_path
        if tokenizer is None:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def __getitem__(self, idx):
        return self.memory_thread[idx]

    def __len__(self):
        return self.memory_thread.shape[0]

    def save(self, path: Union[str,None]) -> None:
        if path is None:
            path = os.path.join(self.save_path,f'{self.name}.parquet')
        self.memory_thread.write_parquet(
            file=path,
            compression='zstd',
            compression_level=None,
            statistics=False,
            row_group_size=None,
            use_pyarrow=True,
            pyarrow_options=None
        )

    def load(self, path: Union[str,None]) -> None:
        if path is None:
            path = os.path.join(self.save_path,f'{self.name}.parquet')
        self.memory_thread = pl.read_parquet(source = path,
                        use_pyarrow= True,
                        memory_map = True,
                        )
        
    def get_total_tokens_from_thread(self):
        return self.memory_thread["tokens_count"].sum()

    def reset_memory(self) -> None:
        self.memory_thread = pl.DataFrame(schema=self.memory_schema) 

    def dict_to_row(self, message_dict:Dict[str,str]) -> pl.DataFrame:
        timestamp = message_dict['timestamp'] if 'timestamp' in message_dict else [now()]
        return pl.DataFrame(schema=self.memory_schema,
                            data={ "role":[message_dict["role"]],
                                  "content":[message_dict["content"]],
                                  "timestamp": timestamp,
                                  "tokens_count":[len(self.tokenizer.encode(message_dict["content"]))+7]
                                  })

    def get_message_tokens_from_dict(self, message_dict: dict) -> int:
        """
        Calculate the number of tokens in a message, including the role token.

        :param message_dict: A dictionary containing the role and content of the message.
        :return: The total number of tokens in the message.
        """
        message_dict = check_dict(message_dict)
        message = message_dict["content"]
        return len(self.tokenizer.encode(message)) + 7  # +7 for the role token

    def get_message_role_from_dict(self, message_dict: dict) -> str:
        """
        Get the role of the message from a message dictionary.

        :param message_dict: A dictionary containing the role and content of the message.
        :return: The role of the message.
        """
        message_dict = check_dict(message_dict)
        return message_dict["role"]

    def add_dict_to_thread(self, message_dict: dict) -> None:
        """
        Add a message to the memory thread.

        :param message_dict: A dictionary containing the role and content of the message.
        """
        
        new_message_row = self.dict_to_row(message_dict)

        if (
            self.max_memory is None
            or self.total_tokens + new_message_row['tokens_count'] <= self.max_memory
        ):
            self.memory_thread = pl.concat([self.memory_thread, new_message_row], rechunk=True)
            self.total_tokens = self.get_total_tokens_from_thread()
        else:
            print("The memory BaseThread is full, the last message was not added")
            

    def remove_dict_from_thread(
        self, message_dict: Union[Dict, None] = None, idx: Union[int, None] = None
    ) -> None:
        
        if message_dict is None and idx is None:
            raise Exception("You need to provide either a message_dict or an idx")

        elif idx is not None and idx < len(self.memory_thread):
            
            self.memory_thread = self.memory_thread.lazy().with_row_count("index").filter(pl.col("index") != idx).drop("index").collect()
            self.total_tokens = self.get_total_tokens_from_thread() 
            return

        elif message_dict is not None:
            message_dict = check_dict(message_dict)
            self.memory_thread = self.memory_thread.lazy().filter(self.memory_thread["content"] != message_dict["content"]).collect()
            self.total_tokens = self.get_total_tokens_from_thread()
            return

        else:
            raise Exception("Index was out bound and no corresponding content found.")


    def find_message(
        self, message: Union[dict, str]
        ) -> pl.DataFrame:
        """
        Find a message in the memory thread. If the message is a dictionary, it will search for the exact match.
        If the message is a string, it will search for the string in the content of the message dictionary."""

        message = message if isinstance(message, str) else check_dict(message)
        
        if isinstance(message,str):
            return self.memory_thread.lazy().filter(pl.col("content") == message).collect()

        else:
            return self.memory_thread.lazy().filter((pl.col("content") == message["content"]) & (pl.col("role") == message['role'])).collect()


        

    def last_message(self, role: Union[str, None] = None) ->  pl.DataFrame:
        """
        Get the last message in the memory thread with a specific role."""
        if role is None:
            return self.memory_thread[-1]
        else:
            return self.memory_thread.lazy().filter(pl.col("role") == role).collect()[-1]

    def first_message(self, role: Union[str, None] = None) -> pl.DataFrame: 
        """
        Get the first message in the memory thread with a specific role."""
        if role is None:
            return self.memory_thread[0]
        else:
            return self.memory_thread.lazy().filter(pl.col("role") == role).collect()[0]


    def messages_before( self, message: dict   ) -> pl.DataFrame:
        """
        Get all messages before a specific message in the memory thread."""
        
        index = self.memory_thread.lazy().with_row_count("index").filter((pl.col('content')==message['content']) & (pl.col('role')==message['role'])).select('index').collect()[0][0]
        return self.memory_thread.lazy().with_row_count("index").filter(pl.col('index')<index).collect()
    
    def messages_after( self, message: dict   ) -> pl.DataFrame:
        """
        Get all messages after a specific message in the memory thread."""
        
        index = self.memory_thread.lazy().with_row_count("index").filter((pl.col('content')==message['content']) & (pl.col('role')==message['role'])).select('index').collect()[0][0]
        return self.memory_thread.lazy().with_row_count("index").filter(pl.col('index')>index).collect()

    def messages_between(
        self, start_message: dict, end_message: dict ) -> pl.DataFrame:
        """
        Get all messages between two specific messages in the memory thread with a specific role."""
        start_index = self.memory_thread.lazy().with_row_count("index").filter((pl.col('content')==start_message['content']) & (pl.col('role')==start_message['role'])).select('index').collect()[0][0]
        end_index = self.memory_thread.lazy().with_row_count("index").filter((pl.col('content')==end_message['content']) & (pl.col('role')==end_message['role'])).select('index').collect()[0][0]
        return self.memory_thread.lazy().with_row_count("index").filter((pl.col('index')<end_index) & (pl.col('index'))>start_index).collect()

    def messages_more_tokens(self, tokens: int, role: Union[str, None] = None) -> pl.DataFrame:
        """
        Get all messages with more tokens than a specific number in the memory thread with a specific role."""
        
        return self.memory_thread.lazy().filter((pl.col('role')==role) & (pl.col('tokens_count'))>tokens).collect()

    def messages_less_tokens(self, tokens: int, role: Union[str, None] = None) -> pl.DataFrame:
        """
        Get all messages with less tokens than a specific number in the memory thread with a specific role."""
        
        return self.memory_thread.lazy().filter((pl.col('role')==role) & (pl.col('tokens_count'))<tokens).collect()


    def messages_after_time(self, timestamp: int, role: Union[str, None] = None) -> pl.DataFrame:
        """
        Get all messages with more tokens than a specific number in the memory thread with a specific role."""
        
        return self.memory_thread.lazy().filter((pl.col('role')==role) & (pl.col('timestamp')>timestamp)).collect()

    def messages_before_time(self, timestamp: int, role: Union[str, None] = None) -> pl.DataFrame:
        """
        Get all messages with more tokens than a specific number in the memory thread with a specific role."""
        
        return self.memory_thread.lazy().filter((pl.col('role')==role) & (pl.col('timestamp')<timestamp)).collect()

    def select_col(self, feature: Union[List[str],str]):
        return self.memory_thread[feature]

    def filter_col(self, feature: str, filter: str):
        try:
            return self.memory_thread.lazy().filter(pl.col(feature) == filter).collect()
        except Exception as e:
            return str(e)



    def token_bound_history(
        self, max_tokens: int, role: Union[str, None] = None
    ):

        reversed_df = self.memory_thread.lazy().with_row_count("index").reverse()   
        reversed_df = reversed_df.with_columns(pl.col("tokens_count").cumsum().alias("cum_tokens_count"))  
        filtered_df = reversed_df.filter((pl.col("cum_tokens_count") <= max_tokens) & (pl.col("role") != role)).collect()
        messages = filtered_df['content']
        indexes = filtered_df['index']

        return messages,indexes

    
    def load_from_gpt_url(self, url: str):

        response = requests.get(url)
        if response.status_code == 200:
        
            soup = BeautifulSoup(response.text, 'html.parser')
        else:
            raise ValueError(f"Non è stato possibile accedere alla pagina. Codice di stato: {response.status_code}")
            
        
        next_data = soup.find('script', id='__NEXT_DATA__')

        if next_data is not None:

            data_string = next_data.string  # pyright: ignore 

            json_obj = json.loads(data_string)  # pyright: ignore

            conversation_data = json_obj['props']['pageProps']['serverResponse']['data']

            messages = conversation_data['mapping']

            for _,value in messages.items():
                if ('parent' in value.keys()) and (value['message']['content']['parts'][0] != ''):
                    message_dict = {'role':value['message']['author']['role'],
                                    'content': value['message']['content']['parts'][0],
                                    'timestamp': value['message']['create_time']}
                    self.add_dict_to_thread(message_dict)
            print("Conversation loaded succesfully")
        else:
            raise ValueError(f"Nessuna conversazione trovata a questo link!")



        

