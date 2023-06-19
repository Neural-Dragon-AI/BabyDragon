# BabyDragon

```
python -m pip install --editable path_to\BabyDragon
```
* re-factoring thread to polar
* re-factor memory index to numpy
* add polar queries to filter memory search
* re-write prompting to be a combination of f-strings and polar queries 


# @Daniel
## Tasks from new gpt overlord

Hi Daniel,

Welcome to the BabyDragon project! This project is designed to provide a structured, scalable, and efficient memory indexing system. Here's a brief introduction and guide to get you started.

## Project Structure

The project primarily revolves around three classes, namely `BaseIndex`, `NpIndex` and `MemoryIndex`.

- `BaseIndex`: This is the abstract base class that defines the necessary methods an index class should have. Check it out [here](https://github.com/Neural-Dragon-AI/BabyDragon/blob/polars/babydragon/memory/indexes/base_index.py).
- `NpIndex`: This class inherits from `BaseIndex` and provides an index structure based on numpy arrays. You can find more information about it [here](https://github.com/Neural-Dragon-AI/BabyDragon/blob/polars/babydragon/memory/indexes/numpy_index.py).
- `MemoryIndex`: This class, which is a subclass of `NpIndex`, introduces a new concept of 'context' that creates a one-to-many mapping from values to context. The context is essentially a list of lists, where the same value could be present in multiple rows of a frame. The class is about 50% complete and requires further development and testing. Check out the current version [here](https://github.com/Neural-Dragon-AI/BabyDragon/blob/polars/babydragon/memory/indexes/memory_index.py).

## Utilities

The `dataframes.py` script contains several utility functions which are used for handling and manipulating dataframes in this project. You can see the existing utilities [here](https://github.com/Neural-Dragon-AI/BabyDragon/blob/polars/babydragon/utils/dataframes.py). These utilities need to be fully implemented and tested.

## Your Tasks

- Implement the utility functions in `dataframes.py`.
- Implement the function `extract_values_and_embeddings_python` in `pythonparser.py`. This function should extract values and their corresponding embeddings from Python code. The file can be found [here](https://github.com/Neural-Dragon-AI/BabyDragon/blob/polars/babydragon/utils/pythonparser.py).
- Continue the development of `MemoryIndex`, particularly the logic for loading from frames.
- Test all the classes and functions you've worked on.

Once you've implemented `extract_values_and_embeddings_python`, start reasoning about the meaning of a concept in an index of Python values. This will be crucial for the later stages of the project.

Best of luck!

Your friend,
[Your Name]
