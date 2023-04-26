from typing import List, Tuple

import numpy as np


def softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def group_items_by_rank_buckets(
    code_strings: List[str],
    node_embeddings: np.ndarray,
    S_vectors: np.ndarray,
    S_vectors_type: str,
    component_window_size: int = 5,
    use_softmax: bool = True,
) -> List[Tuple[List[str], np.ndarray]]:
    if S_vectors_type not in ("U", "Vt"):
        raise ValueError("Invalid S_vectors_type value. It must be either 'U' or 'Vt'.")

    if use_softmax:
        S_vectors = softmax(S_vectors, axis=1)

    num_buckets = S_vectors.shape[0] // component_window_size
    rank_buckets = []

    for i in range(num_buckets):
        start_idx = i * component_window_size
        end_idx = (i + 1) * component_window_size

        if S_vectors_type == "U":
            target_matrix = S_vectors[start_idx:end_idx, :]
            contributions = np.sum(np.abs(target_matrix), axis=0)
        elif S_vectors_type == "Vt":
            target_matrix = S_vectors[:, start_idx:end_idx]
            contributions = np.sum(np.abs(target_matrix), axis=1)

        indexes = np.argsort(contributions)[::-1]

        # Filter out any indexes that are out of range
        indexes = [idx for idx in indexes if idx < len(code_strings)]

        sorted_code_strings = [code_strings[idx] for idx in indexes]
        sorted_node_embeddings = node_embeddings[indexes]

        rank_buckets.append((sorted_code_strings, sorted_node_embeddings))

    return rank_buckets
