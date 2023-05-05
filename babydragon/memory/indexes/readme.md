# BabyDragon Indexes

The `indexes` submodule of the BabyDragon package provides different indexing
and searching strategies for various data types.
The main class in this
submodule is  `MemoryIndex` class, a wrapper for a Faiss index that simplifies managing the index and associated data. It supports creating an index from scratch, loading an index from a file, or initializing from a pandas DataFrame. The class also provides methods for adding and removing items from the index, querying the index, saving and loading the index, and pruning the index based on certain constraints.

##  Table of Contents

1. [MemoryIndex](#usage)
   - [Initializing a MemoryIndex](#initializing-a-memoryindex)
   - [Adding and Removing Items](#adding-and-removing-items)
   - [Querying the Index](#querying-the-index)
   - [Saving and Loading](#saving-and-loading)
   - [Pruning the Index](#pruning-the-index)
   - [Multithreading](#multithreading)
2. [Examples](#examples)

## Usage

### Initializing a MemoryIndex

A `MemoryIndex` object can be initialized in several ways:

1. Create a new empty index from scratch:

```python
from babydragon.indexes import MemoryIndex

index = MemoryIndex()
```
2. Create a new index from a list of values:

```python
values = ["apple", "banana", "cherry"]

index = MemoryIndex(values=values)
```
3. Create a new index from a list of values and their embeddings:
```python
values = ["apple", "banana", "cherry"]
embeddings = [...]  # list of embeddings corresponding to the values

index = MemoryIndex(values=values, embeddings=embeddings)
```

4. Create a new index from a list of values and their embeddings:
```python
values = ["apple", "banana", "cherry"]
embeddings = [...]  # list of embeddings corresponding to the values

index = MemoryIndex(values=values, embeddings=embeddings)
```

5. Load an existing index from a file:
```python
index = MemoryIndex(load=True, name: "precomputed_index")
```
6. Initialize a MemoryIndex object from a pandas DataFrame:
```python
import pandas as pd

data_frame = pd.DataFrame({
    "values": ["apple", "banana", "cherry"],
    "embeddings": [...]  # list of embeddings corresponding to the values
})

index = MemoryIndex.from_pandas(data_frame=data_frame, columns="values", embeddings_col="embeddings")
```


### Adding and Removing Items
You can add items to the index by calling the add_to_index method:
```python
index.add_to_index(value="orange")
```
You can also remove items from the index by calling the remove_from_index method:


```python
index.remove_from_index(value="banana")
```
### Querying the Index
To query the index, use the faiss_query or token_bound_query methods:

```python
# Query the top-5 most similar values
values, scores, indices = index.faiss_query(query="fruit", k=5)

# Query the top-5 most similar values with a maximum tokens constraint
values, scores, indices = index.token_bound_query(query="fruit", k=5, max_tokens=4000)
```
### Saving and Loading
You can save the index to a file by calling the save method:

```python
index.save()
```
You can load an index from a file by calling the load method:

```python
index = MemoryIndex(load=True, name= "precomputed_index")
```

### Pruning the Index
To prune the index based on certain constraints, use the prune method:
```python
index.prune(max_tokens=3500)
```
### Multithreading
In order to enable multi-threading for speeding up the embedding process, you can set the `max_workers parameter to a value bigger than 1:
```python
index = MemoryIndex(values=myvalues,max_workers=8)
```

## Subclasses

### PandasIndex

`PandasIndex` is a subclass of `MemoryIndex` that creates an index object from a
pandas DataFrame. It is useful for indexing and searching data stored in pandas
DataFrames.

### PythonIndex

`PythonIndex` is another subclass of `MemoryIndex` that creates an index object
from a directory of Python source code files. It inherits from both
`MemoryIndex` and `PythonParser`. It is useful for indexing and searching Python
source code files.

### PythonIndex Example

```python
from babydragon.memory.indexes import PythonIndex

# Create a PythonIndex instance for a directory of Python source code files
index = PythonIndex(directory="path/to/python/files", name: "mypythonindex" )

# Add a new Python file to the index
index.add_to_index("new_python_file.py")

# Search for similar code snippets using a query
query = "def function_example():"
search_results = index.faiss_query(query, top_k=3)

print("Search results:")
for item, score in search_results:
    print(f"Item: {item}, Score: {score}")

# Save the index to a file
index.save()

# Load the index from a file
loaded_index = PythonIndex(load=True,  name: "mypythonindex")

# Perform a search on the loaded index
search_results_loaded = loaded_index.faiss_query(query, top_k=3)

print("Search results (loaded index):")
for item, score in search_results_loaded:
    print(f"Item: {item}, Score: {score}")
```

