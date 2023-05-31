multi_kernel_task
=================

.. code-block:: python

	
	
	class MultiKernelTask(BaseTask):
	    def __init__(
	        self,
	        memory_kernel_dict: Dict,
	        chatbot: BaseChat,
	        parent_kernel_label: str,
	        child_kernel_label: str,
	        system_prompt: str,
	        clustering_method: str,
	        path_group: Dict[str, List[List[int]]],
	        task_id: str = "MultiKernelTask",
	        max_workers: int = 1,
	        calls_per_minute: int = 20,
	    ):
	        self.clustering_method = clustering_method
	        self.parent_kernel_label = parent_kernel_label
	        self.child_kernel_label = child_kernel_label
	        self.memory_kernel_dict = memory_kernel_dict
	        
	        self._setup_memory_kernel_group()
	        if path_group:
	            self.memory_kernel_group.path_group = path_group
	        else:
	            self.generate_task_paths()
	        self.system_prompt = system_prompt
	        self.chatbot = chatbot
	        self.paths = self.memory_kernel_group.path_group[self.parent_kernel_label]
	        super().__init__(path = self.paths, max_workers=max_workers, task_id=task_id, calls_per_minute=calls_per_minute)
	
	
	    def _setup_memory_kernel_group(self):
	        if self.clustering_method == "HDBSCAN":
	            print("Using HDBSCAN")
	            self.memory_kernel_group = HDBSCANMultiKernel(memory_kernel_dict=self.memory_kernel_dict)
	        elif self.clustering_method == "Spectral":
	            print("Using Spectral")
	            self.memory_kernel_group = SpectralClusteringMultiKernel(memory_kernel_dict=self.memory_kernel_dict)
	        else:
	            raise ValueError(f"Unknown clustering method: {self.clustering_method}")
	
	    def generate_task_paths(self):
	        print("Generating task paths")
	
	        self.memory_kernel_group.generate_path_groups()
	
	    def llm_response(self, chatbot: BaseChat, message: str, context=None, id=None):
	        max_tokens = 8000 if chatbot.model == "gpt-4" else 4000
	        return chatbot.reply(message)
	
	    def _execute_sub_task(self, sub_path) -> List[str]:
	        if self.parallel:
	            chatbot_instance = copy.deepcopy(self.chatbot)
	        else:
	            chatbot_instance = self.chatbot
	
	        sub_results = {}
	        for i in sub_path:
	            print(f'Current_node: {i}, size of values {len(self.memory_kernel_group.memory_kernel_dict[self.parent_kernel_label].values)}')
	            try:
	                current_val = self.memory_kernel_group.memory_kernel_dict[self.parent_kernel_label].values[i]
	                response = self.llm_response(chatbot_instance, current_val, id=i)
	                sub_results[i] = response
	            except IndexError:
	                print(f"Error: Invalid index {i} in sub_path")
	                sub_results[i] = f"Error: Invalid index {i} in sub_path"
	            except Exception as e:
	                print(f"Error in sub_task for index {i}: {e}")
	                sub_results[i] = f"Error in sub_task for index {i}: {e}"
	
	        return sub_results
	
	    def execute_task(self) -> None:
	        BaseTask.execute_task(self)
	
	        # Load the results from the JSON file
	        # with open(f"{self.task_id}_results.json", "r") as f:
	        #     task_results = json.load(f)
	        self._load_results_from_file()
	        task_results = self.results
	        new_values = []
	        #sort task_results by index and add to new_values 0- max values ascending
	        for task_result in task_results:
	            if isinstance(task_result, dict):
	                for key, value in task_result.items():
	                    new_values.append((int(key), value))
	            elif isinstance(task_result, str):
	                print(f"Error in task_result: {task_result}")
	
	        new_values.sort(key=lambda x: x[0])
	        values = [x[1] for x in new_values]
	
	        task_memory_index = MemoryIndex()
	        task_memory_index.init_index(values=values)
	        # Create a new MemoryKernel with the results
	        new_memory_kernel = MemoryKernel.from_task_results(task_memory_index)
	
	        # Add the new MemoryKernel to the MultiKernel
	        self.memory_kernel_group.memory_kernel_dict[self.child_kernel_label] = new_memory_kernel
	        self.generate_task_paths()
	
	        #delete the results file
	        # os.remove(f"{self.task_id}_results.json")
	

.. automodule:: multi_kernel_task
   :members:
