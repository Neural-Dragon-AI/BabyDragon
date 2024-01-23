#### Simulated Polars DataFrame for Graph Representation
```python
import polars as pl

# Sample input data
data = {
    'node_id': [1, 2, 3, 4, 5],
    'edges': [[2, 3], [1, 4], [1, 5], [2, 5], [3, 4]]
}

GraphDataFrame = pl.DataFrame(data)
print(GraphDataFrame.head(5))

shape: (5, 2)
┌─────────┬─────────┐
│ node_id ┆ edges   │
│ ---     ┆ ---     │
│ i64     ┆ list[i] │
╞═════════╪═════════╡
│ 1       ┆ [2, 3]  │
│ 2       ┆ [1, 4]  │
│ 3       ┆ [1, 5]  │
│ 4       ┆ [2, 5]  │
│ 5       ┆ [3, 4]  │
└─────────┴─────────┘
```
```python

# Degree of each node
degree_series = GraphDataFrame.graph.node_metrics.degree()
print(degree_series.head(5))

# Output
shape: (5, 1)
┌────────┐
│ degree │
│ ---    │
│ i64    │
╞════════╡
│ 2      │
│ 2      │
│ 2      │
│ 2      │
│ 2      │
└────────┘
```
```python
# Eigenvector centrality of each node
centrality_series = GraphDataFrame.graph.node_metrics.eigenvector_centrality()
print(centrality_series.head(5))

# Output
shape: (5, 1)
┌────────────────────┐
│ eigenvector_centr  │
│ ---                │
│ f64                │
╞════════════════════╡
│ 0.5                │
│ 0.6                │
│ 0.4                │
│ 0.7                │
│ 0.5                │
└────────────────────┘

```

```python
# Shortest path distances from all nodes to node_id 3, with path represented as list of node IDs
distances = GraphDataFrame.graph.distances.short_path(3, algorithm='dijkstra', create_path_column=True)
print(distances.head(5))

# Output
shape: (5, 3)
┌─────────┬──────────┬────────────┐
│ node_id ┆ distance ┆ path       │
│ ---     ┆ ---      ┆ ---        │
│ i64     ┆ i64      ┆ list[i64]  │
╞═════════╪══════════╪════════════╡
│ 1       ┆ 1        ┆ [1, 3]     │
│ 2       ┆ 1        ┆ [2, 3]     │
│ 3       ┆ 0        ┆ [3]        │
│ 4       ┆ 2        ┆ [4, 2, 3]  │
│ 5       ┆ 1        ┆ [5, 3]     │
└─────────┴──────────┴────────────┘
```
```python
# Shortest path distances from all nodes to node_ids 3 and 4
targets = [3, 4]
distances = GraphDataFrame.graph.distances.short_path(targets, algorithm='dijkstra', create_path_column=True)
print(distances.head(5))

# Output
shape: (5, 5)
┌─────────┬─────────────────┬───────────────┬─────────────────┬───────────────┐
│ node_id ┆ distance_to_3   ┆ path_to_3     ┆ distance_to_4   ┆ path_to_4     │
│ ---     ┆ ---             ┆ ---           ┆ ---             ┆ ---           │
│ i64     ┆ i64             ┆ list[i64]     ┆ i64             ┆ list[i64]     │
╞═════════╪═════════════════╪═══════════════╪═════════════════╪═══════════════╡
│ 1       ┆ 1               ┆ [1, 3]        ┆ 2               ┆ [1, 2, 4]     │
│ 2       ┆ 1               ┆ [2, 3]        ┆ 1               ┆ [2, 4]        │
│ 3       ┆ 0               ┆ [3]           ┆ 1               ┆ [3, 4]        │
│ 4       ┆ 1               ┆ [4, 2, 3]     ┆ 0               ┆ [4]           │
│ 5       ┆ 1               ┆ [5, 3]        ┆ 1               ┆ [5, 4]        │
└─────────┴─────────────────┴───────────────┴─────────────────┴───────────────┘


```
```python
# Grouping based on node clusters and aggregating
clustered_grouping = GraphDataFrame.graph_group_by.aggolmerative_clustering(num_clusters=2, depth=1).agg([
    pl.col("cluster"),
    pl.col("edges").list().alias("connected_nodes")
])
print(clustered_grouping.head(5))

# Output
shape: (2, 2)
┌─────────┬──────────────────────────┐
│ cluster ┆ connected_nodes          │
│ ---     ┆ ---                      │
│ i64     ┆ list[list[i]]            │
╞═════════╪══════════════════════════╡
│ 0       ┆ [[2, 3], [1, 4]]         │
│ 1       ┆ [[1, 5], [2, 5], [3, 4]] │
└─────────┴──────────────────────────┘


```

