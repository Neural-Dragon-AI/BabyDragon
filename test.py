from babydragon.chat.memory_chat import FifoVectorChat, FifoChat, VectorChat
from babydragon.chat.base_chat import BaseChat
from babydragon.chat.chat import Chat
from babydragon.memory.indexes.pandas_index import PandasIndex
from babydragon.working_memory.short_term_memory.git_memory import GitMemory
import os
import openai

if __name__ == '__main__':
    openai.api_key = "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"

    username = "Danielpatrickhug"
    repo_name = "GitModel"
    base_directory = "work_folder"

    # Make sure the work folder exists
    if not os.path.exists(base_directory):
        os.mkdir(base_directory)

    git_memory = GitMemory(username, repo_name)
    git_memory.create_indexes(base_directory)

    print(git_memory)

