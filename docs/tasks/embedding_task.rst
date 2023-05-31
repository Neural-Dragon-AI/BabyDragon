embedding_task
==============

.. code-block:: python

	
	
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
	

.. automodule:: embedding_task
   :members:
