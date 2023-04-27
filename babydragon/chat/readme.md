# BabyDragon Chatbot Submodule

This submodule provides a set of chatbot classes that leverage different memory thread strategies for managing the context provided to the model. The chatbot subclasses are designed to be flexible and extensible, allowing users to customize the chatbot's behavior. The chatbot classes inherit from the base classes and combine them with the other modules to build more advanced and specialized chatbots.

## Subclasses and Functionalities

### FifoChat

`FifoChat` inherits from `FifoThread` and `Chat`. It implements a chatbot that uses a First-In-First-Out (FIFO) memory management strategy. The oldest messages are removed first when reaching the `max_fifo_memory` limit.

#### Key Methods

- `fifo_memory_prompt`: Composes the prompt for the model, including the system prompt and memory thread.
- `query`: Queries the chatbot with a given question, adds the question and answer to the memory, and returns the chatbot's response.

### VectorChat

`VectorChat` inherits from `VectorThread` and `Chat`. It implements a chatbot that uses a vector-based memory management strategy. The memory prompt is constructed by filling the memory with the `k` most similar messages to the question until the `max_vector_memory` limit is reached.

#### Key Methods

- `vector_memory_prompt`: Combines the system prompt, `k` most similar messages to the question, and the user prompt.
- `weighted_memory_prompt`: Combines the system prompt, weighted by temporal decay `k` most similar messages to the question, and the user prompt.
- `query`: Queries the chatbot with a given question, adds the question and answer to the memory, and returns the chatbot's response.

### FifoVectorChat

`FifoVectorChat` inherits from `FifoThread` and `Chat`. It implements a chatbot that combines both FIFO and Vector memory strategies. The memory prompt is constructed by including both FIFO memory and Vector memory.

#### Key Methods

- `setup_longterm_memory`: Sets up long-term memory by allocating memory for the FIFO and Vector memory components.
- `fifovector_memory_prompt`: Combines the system prompt, long-term memory (vector memory), short-term memory (FIFO memory), and the user prompt.
- `query`: Queries the chatbot with a given question, adds the question and answer to the memory, and returns the chatbot's response.

## Usage

To create a chatbot instance, simply import the desired chatbot class and instantiate it with the appropriate parameters. For example:

```python
from babydragon.chat.submodule import FifoChat

chatbot = FifoChat(model="gpt-4", max_fifo_memory=2048, max_output_tokens=1000)
```
You can then query the chatbot using the query method:
```python

response = chatbot.query("What is the capital of France?")
print(response)
```

## Creating and Using Chatbots with Memory Indexes

This section provides examples of how to create chatbots with and without a memory index and instantiate each type of memory with appropriate initialization parameters.

### Chatbot without Memory Index

You can create a chatbot without a memory index by simply initializing it with the default parameters. For example, using the `FifoChat` class:

```python
from babydragon.chat.submodule import FifoChat

chatbot_no_index = FifoChat(model="gpt-4", max_fifo_memory=2048, max_output_tokens=1000)
response = chatbot_no_index.query("What is the capital of France?")
print(response)
```

### Chatbot with Memory Index
To create a chatbot with a memory index, you need to initialize the memory index and pass it to the chatbot. The following example demonstrates how to create a PandasIndex and use it with a VectorChat chatbot:

```python
from babydragon.memory.indexes.pandas_index import PandasIndex
from babydragon.chat.submodule import VectorChat

# Initialize the PandasIndex
index = PandasIndex(data_path="path/to/data.csv")

# Create a dictionary containing the index
index_dict = {'my_index': index}

# Instantiate the VectorChat chatbot with the memory index
chatbot_with_index = FifoChat(model="gpt-4",index_dict = index_dict, max_fifo_memory=2048, max_output_tokens=1000, max_index_memory = 500)

# Query the chatbot
response = chatbot_with_index.query("What is the capital of France?")
print(response)
```

## Creating Each Type of Memory
You can initialize each type of memory by providing the appropriate parameters during instantiation. Here are some examples for each type of memory:

### FifoChat
```python

from babydragon.chat.submodule import FifoChat

fifo_chatbot = FifoChat(model="gpt-4", max_fifo_memory=2048, max_output_tokens=1000)
```
### VectorChat
```python

from babydragon.chat.submodule import VectorChat

vector_chatbot = VectorChat(model="gpt-4", max_vector_memory=2048, max_output_tokens=1000)
```
### FifoVectorChat
```python

from babydragon.chat.submodule import FifoVectorChat

fifovector_chatbot = FifoVectorChat(model="gpt-4", max_memory=4096, max_output_tokens=1000, longterm_frac=0.5)
```

