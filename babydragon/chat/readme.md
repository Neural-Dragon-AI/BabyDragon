# BabyDragon Chatbot Submodule

This submodule provides a set of chatbot classes that leverage different memory thread strategies for managing the context provided to the model. The classes are designed to be flexible and extensible, allowing users to customize the chatbot's behavior. The chatbot classes inherit from the base classes and combine them with the other modules to build more advanced and specialized chatbots.

## Base Classes

### BaseThread

This class represents a basic memory thread that can store messages and manage memory.

### BaseChat

A chatbot base class that manages the user and system prompts and serves as the interface for querying the model.

### Prompter

An interface for composing prompts based on different memory thread strategies.

## Inheritance Structure

The chatbot classes in this submodule inherit from the base classes and combine them with the other modules as follows:

- `FifoChat`: Inherits from `FifoThread` and `Chat`
- `VectorChat`: Inherits from `VectorThread` and `Chat`
- `FifoVectorChat`: Inherits from `FifoThread` and `Chat`

## Integration with Other Modules

The chatbot classes in this submodule interact with other modules as follows:

- Memory Threads: The chatbot classes use memory threads like `FifoThread`, `VectorThread`, and combinations of them to manage the context provided to the model.
- Indexes: The chatbot classes can use indexes like `PandasIndex` and `MemoryIndex` for searching and sorting messages.
- Utils: Utility functions like `mark_question`, `mark_system`, and `mark_answer` are used to label and format messages.

## Usage

To create a chatbot instance, simply import the desired chatbot class and instantiate it with the appropriate parameters. For example:

```python
from babydragon.chat.submodule import FifoChat

chatbot = FifoChat(model="gpt-4", max_fifo_memory=2048, max_output_tokens=1000)
