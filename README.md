# BabyDragon :dragon:

## Introduction
Welcome to BabyDragon, a comprehensive and rigorously designed Python package that aims to be the go-to toolbox for AI researchers, ML Engineers, and Python developers. Whether you need to interact with chat models, manipulate dataframes, analyze Python code, or even access GitHub repositories for code analysis, BabyDragon has got you covered.

## Features :nut_and_bolt:
- `Abstract Base Classes`: BaseFrame and BaseThread provides the architecture for custom data frames/threads with attributes like tokenizers, meta columns, etc.
- `Code Data Management`: CodeFrame allows for operations such as tokenization, validation, embedding, summarization, LIBCST Parsing, and querying of code data(SQL + dot product similarity).
- `Asynchronous GPT-3 Batch Processing`: PolarsGenerator for batch generation using OpenAI's GPT-3.
- `Chat Data Analysis`: ChatFrame offers robust tools for handling chat data including message tokenization, summarization, embedding, clustering, and searching.
- `Chatbot Creation and Memory Management`: Leverage BaseThread and ChatFrame for creating chatbots with durable conversation memory.
- `Memory-based DataFrame`: MemoryFrame for in-memory data manipulation and analysis.
- `Status Tracking`: Models to keep tabs on API requests and their statuses.
- `Utility Functions`: Helper functions to extract and manipulate data from Hugging Face datasets.

