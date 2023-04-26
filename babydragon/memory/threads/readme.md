# BabyDragon Threads

BabyDragon Threads is a module within the BabyDragon AI project that provides memory structures for storing, managing, and retrieving information. The main components of this module are the `BaseThread` class, and its subclasses, `FifoThread` and `VectorThread`.

## BaseThread

`BaseThread` is the base class for memory structures in BabyDragon. It provides basic methods for adding, removing, and managing messages in a memory thread.

### Subclasses

1. **FifoThread**: A memory structure following the First-In-First-Out (FIFO) principle. When the memory reaches its maximum capacity, the oldest messages are removed first. It can also store redundant messages in a separate thread called `lucid_memory` and pass important messages to the `longterm_memory`.

2. **VectorThread**: A memory structure that uses Faiss, an efficient similarity search, and clustering library, to create a vector index for stored messages. This enables efficient searching for similar messages in the memory based on a query.

## Examples

### FifoThread

```python
from babydragon.memory.threads.base_thread import BaseThread
from babydragon.memory.threads.fifo_thread import FifoThread

# Initialize the FifoThread
fifo_memory = FifoThread(name='fifo_memory', max_memory=1000, longterm_thread=None, redundant=True)

# Add a message to the memory
message_dict = {'role': 'user', 'content': 'Hello, world!'}
fifo_memory.add_message(message_dict)
```
### VectorThread
```python

from babydragon.memory.threads.base_thread import BaseThread
from babydragon.memory.threads.vector_thread import VectorThread

# Initialize the VectorThread
vector_memory = VectorThread(name='vector_memory', max_context=2048, use_mark=False)

# Add a message to the memory
message_dict = {'role': 'user', 'content': 'Hello, world!'}
vector_memory.add_message(message_dict)

# Query the memory
query = "What's the weather like today?"
results, scores, indices = vector_memory.sorted_query(query, k=10, max_tokens=4000)
```

### Combining FifoThread and VectorThread

You can combine both `FifoThread` and `VectorThread` to create a memory structure that benefits from the features of both classes by specifying the `VectorThread` as the long-term memory thread for the `FifoThread`.

```python
from babydragon.memory.threads.base_thread import BaseThread
from babydragon.memory.threads.fifo_thread import FifoThread
from babydragon.memory.threads.vector_thread import VectorThread

# Initialize the VectorThread
vector_memory = VectorThread(name='vector_memory', max_context=2048, use_mark=False)

# Initialize the FifoThread with VectorThread as long-term memory thread
fifo_memory = FifoThread(name='fifo_memory', max_memory=10, longterm_thread=vector_memory, redundant=True)

# Add messages to the FifoThread
messages = [
    {'role': 'user', 'content': 'What is the capital of France?'},
    {'role': 'assistant', 'content': 'The capital of France is Paris.'},
    {'role': 'user', 'content': 'What is the population of Paris?'},
    {'role': 'assistant', 'content': 'The population of Paris is approximately 2.1 million people.'},
]

for message in messages:
    fifo_memory.add_message(message)

# Query the VectorThread
query = "What is the capital of France?"
results, scores, indices = vector_memory.sorted_query(query, k=5, max_tokens=4000)

print("Query Results:")
for result in results:
    print(result["content"])
```
In this example, FifoThread is initialized with a small memory limit of 10 tokens and the VectorThread is set as its long-term memory thread. As messages are added to FifoThread, any message that exceeds the token limit will be transferred to the VectorThread. When a query is performed on the VectorThread, it retrieves relevant information based on the query. This way, both memory structures are used in tandem to create an AI system that manages memory efficiently and effectively retrieves relevant information when needed.