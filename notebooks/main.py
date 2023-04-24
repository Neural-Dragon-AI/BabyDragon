from babydragon.chat.memory_chat import FifoVectorChat, FifoChat, VectorChat
from babydragon.chat.base_chat import BaseChat
from babydragon.chat.chat import Chat
from babydragon.memory.indexes.pandas_index import PandasIndex

import openai

import pandas as pd
def main():
    openai.api_key = "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"


    #create a mathew colville index
    df_mci = pd.read_csv(r"C:\Users\Tommaso\Documents\Dev\LangDND\DndIndexes\Indexes\matthew_colville_index.csv", converters={"embeddings": eval})
    mci = PandasIndex(df_mci, columns=["text"], embeddings_col="embeddings", name="matthew_colville_index")

    vc = VectorChat(index_dict={"matthew conville": mci})

    print(vc.reply("Dungeon description please"))

if __name__ == "__main__":
    main()