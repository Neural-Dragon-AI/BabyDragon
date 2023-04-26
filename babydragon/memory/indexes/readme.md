# BabyDragon Indexes

The `indexes` submodule of the BabyDragon package provides different indexing and searching strategies for various data types. The main class in this submodule is `MemoryIndex`, which provides the core functionality for indexing and searching. Several subclasses extend this base class to offer specialized solutions for different data types and use cases.

## MemoryIndex

`MemoryIndex` is the base class for all indexes. It is responsible for the core indexing and searching functionality. It allows you to:

- Add items to the index.
- Save and load the index.
- Search for similar items in the index using a query.

## Subclasses

### PandasIndex

`PandasIndex` is a subclass of `MemoryIndex` that creates an index object from a pandas DataFrame. It is useful for indexing and searching data stored in pandas DataFrames.

### PythonIndex

`PythonIndex` is another subclass of `MemoryIndex` that creates an index object from a directory of Python source code files. It inherits from both `MemoryIndex` and `PythonParser`. It is useful for indexing and searching Python source code files.

### MemoryKernel

`MemoryKernel` is a subclass of `MemoryIndex` designed for indexing and searching using graph-based representations of data. It computes the adjacency matrix of a graph based on the similarity of node embeddings and performs k-hop message passing to aggregate information from neighboring nodes. This enables more sophisticated search capabilities based on the relationships between data points.

## Usage

To use any of the index classes, simply import the class from the `babydragon.memory.indexes` submodule and create an instance of the class with the required parameters. For example, to create a `PandasIndex` for a pandas DataFrame, you can do the following:

```python
from babydragon.memory.indexes import PandasIndex
import pandas as pd

data = pd.DataFrame({"text": ["hello", "world", "foo", "bar"]})
index = PandasIndex(pandaframe=data, columns="text")
```

Once you have an instance of an index class, you can add items to the index, save and load the index, and perform search operations using the provided methods.
```python

# Add a new item to the index
index.add_to_index("baz")

# Search for similar items in the index using a query
query = "hello world"
search_results = index.search(query, top_k=3)

print("Search results:")
for item, score in search_results:
    print(f"Item: {item}, Score: {score}")

# Save the index to a file
index.save("example_pandas_index.pkl")

# Load the index from a file
loaded_index = PandasIndex(load=True, save_path="example_pandas_index.pkl")

# Perform a search on the loaded index
search_results_loaded = loaded_index.search(query, top_k=3)

print("Search results (loaded index):")
for item, score in search_results_loaded:
    print(f"Item: {item}, Score: {score}")
```
