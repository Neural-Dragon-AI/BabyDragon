from typing import Any, List

from babydragon.models.embedders.ada2 import OpenAiEmbedder
from babydragon.tasks.base_task import BaseTask


class EmbeddingTask(BaseTask):
    def __init__(
        self,
        embedder: OpenAiEmbedder,
        values: List[Any],
        path: List[List[int]],
        max_workers: int = 1,
        task_id: str = "task",
        calls_per_minute: int = 1500,
        backup: bool = True,
    ):
        BaseTask.__init__(self, path, max_workers, task_id, calls_per_minute, backup)
        self.embedder = embedder
        self.values = values

    def _execute_sub_task(self, sub_path: List[int]) -> List[str]:
        # expected to work with a lig of a single element
        if len(sub_path) != 1:
            raise ValueError(
                "Embedding task expected to work with a list of a single element"
            )
        sub_results = {}
        for i in sub_path:
            embedded_value = self.embedder.embed(self.values[i])
            sub_results[i] = embedded_value
        return sub_results

def parallel_embeddings(embedder, values, max_workers, backup, name):
        # Prepare the paths for the EmbeddingTask
        print("Embedding {} values".format(len(values)))
        paths = [[i] for i in range(len(values))]

        # Initialize the EmbeddingTask and execute it
        embedding_task = EmbeddingTask(
            embedder,
            values,
            path=paths,
            max_workers=max_workers,
            task_id=name + "_embedding_task",
            backup=backup,
        )
        embeddings = embedding_task.work()
        embeddings = [x[1] for x in sorted(embeddings, key=lambda x: x[0])]
        return embeddings