# BabyDragon

```
python -m pip install --editable path_to\BabyDragon
```

- Simply track a history of messages until capacity. Done
- Fifo Queue - Done
- Vector Storage, memory is created by filling the context only with messages
  from the storage,
  - either in Q/A pairs or at message level - Done
  - either ordered in terms of similarity DONE
  - or chronological - TODO
  - summarized in a single message or less messages the {user} token are still
    wasteful (should check if they use only one for that) --> 6 tokens extra per
    message
- Fifo Queue with Outs into VectorStorage, half of the memory is filled with
  samples from the vector storage / half from retrieval.
- Bunch of panda exstension and compositionality

#### IndexChat

The IndexChat is a chatbot that utilizes multiple knowledge indexes for
answering questions. It can be constructed using the following layers:

```
       IndexChat
+---------------------------+
|        Input (Text)       |
+---------------------------+
           |
           v
+---------------------------+
|    Query Embedder         |
+---------------------------+
           |
           v
+---------------------------+
|   Embedded Input          |
+---------------------------+
           |
           v
+---------------------------+
|   Knowledge Index Layer   |
+---------------------------+
           |
           v
+---------------------------+
|    Working Memory         |
+---------------------------+
           |
           v
+---------------------------+
|      Chat-response        |
+---------------------------+
           |
           v
+---------------------------+
|        Output             |
+---------------------------+
```

#### FifoChat

The FifoChat is a simple chatbot that focuses on maintaining a short-term
memory. It can be constructed using the following layers:

- Input Layer: Receives the raw text input from the user.
- Short-term Memory (STM) Layer: Handles the chatbot's short-term memory,
  storing recent interactions in a FIFO (First-In-First-Out) manner.
- Output Layer: Generates a chat-response based on the context provided by the
  short-term memory.

```
       FifoVectorChat
+---------------------------+
|        Input (Text)       |
+---------------------------+
           |
           v
+---------------------------+
|    Short-term Memory (STM)|
+---------------------------+
           |
           v
+---------------------------+
|    Working Memory         |
+---------------------------+
           |
           v
+---------------------------+
|      Chat-response        |
+---------------------------+
           |
           v
+---------------------------+
|        Output             |
+---------------------------+
```

#### VectorChat

The VectorChat is a chatbot that utilizes a long-term memory based on vector
space models. It can be constructed using the following layers:

- Input Layer: Receives the raw text input from the user.
- Query Embedder Layer: Transforms the input text into an embedded
  representation suitable for querying long-term memory.
- Long-term Memory (LTM) Layer: Contains two components:
- LTM Add: Adds new knowledge to long-term memory based on the embedded input.
- LTM Search: Searches long-term memory using the embedded input and returns
  relevant information.
- Output Layer: Generates a chat-response based on the context provided by the
  long-term memory retrieval.

```
       VectorChat
+---------------------------+
|        Input (Text)       |
+---------------------------+
           |
           v
+---------------------------+
|    Query Embedder         |
+---------------------------+
           |
           |
+---------------------------+
|   Embedded Input          |
+---------------------------+
           |
 |ltm_add|   |ltm_search|
           v
+---------------------------+
|   Long-term Memory (LTM)  |
+---------------------------+
           |
           v
+---------------------------+
|    Working Memory
+---------------------------+
           |
           v
+---------------------------+
|      Chat-response        |
+---------------------------+
           |
           v
+---------------------------+
|        Output             |
+---------------------------+
```

#### FifoVectorChat

The FifoVectorChat is a chatbot that combines the short-term memory capabilities
of FifoChat with the long-term memory capabilities of VectorChat. It can be
constructed using the following layers:

- Input Layer: Receives the raw text input from the user.
- Short-term Memory (STM) Layer: Handles the chatbot's short-term memory,
  storing recent interactions in a FIFO (First-In-First-Out) manner.
- Query Embedder Layer: Transforms the input text into an embedded
  representation suitable for querying long-term memory.
- Long-term Memory (LTM) Layer: Contains two components:
- LTM Add: Adds new knowledge to long-term memory based on the embedded input.
- LTM Search: Searches long-term memory using the embedded input and returns
  relevant information.
- Working Memory Layer: Integrates information from both short-term memory and
  long-term memory retrieval to generate a contextualized understanding of the
  conversation. Output Layer: Generates a chat-response based on the context
  provided by the working memory. By constructing these chatbots using the basic
  layers, we can create more complex and specialized chatbot systems that cater
  to various use cases and requirements.

```
       FifoVectorChat
+---------------------------+
|        Input (Text)       |
+---------------------------+
           |
           v
+---------------------------+   embed   +---------------------------+
|    Short-term Memory (STM)|---------->|    Query Embedder         |
+---------------------------+           +---------------------------+
           |                                          |
           |                                          v
           |                             +---------------------------+
           |                             |   Embedded Input          |
           |                             +---------------------------+
           |                                          |
           |                                |ltm_add|   |ltm_search|
           v                                          v
+---------------------------+           +---------------------------+
|    Working Memory         |<----------|   Long-term Memory (LTM)  |
+---------------------------+           +---------------------------+
           |                                          ^
           v                                          |
+---------------------------+                         |
|      Chat-response        |-------------------------|
+---------------------------+
           |
           v
+---------------------------+
|        Output             |
+---------------------------+
```
