git_processor
=============

.. code-block:: python

	
	
	class DirectoryProcessor:
	    def __init__(self, directory_path: str, visitor=FunctionAndClassVisitor()):
	        self.directory_path = directory_path
	        self.visitor = visitor
	
	    def _process_file(self, file_path: str):
	        with open(file_path, "r") as file:
	            source_code = file.read()
	
	        try:
	            tree = cst.parse_module(source_code)
	        except cst.ParserSyntaxError:
	            print(f"Skipping file {file_path}: Failed to parse syntax")
	            return
	
	        tree.visit(self.visitor)
	
	    def process_file(self, file_path: str):
	        # Run flake8 on the file
	        result = subprocess.run(
	            ["flake8", "--select=E999", file_path], capture_output=True
	        )
	
	        if result.returncode != 0:
	            print(f"Skipping file with syntax error: {file_path}")
	            print(result.stderr.decode("utf-8"))
	            return
	
	        with open(file_path, "r") as f:
	            source_code = f.read()
	
	        try:
	            tree = cst.parse_module(source_code)
	            tree.visit(self.visitor)
	        except cst.ParserSyntaxError as e:
	            print(f"Syntax error: {e}")
	            print(f"Skipping file with syntax error: {file_path}")
	
	    def process_directory(self) -> List[str]:
	        function_source_codes = []
	        class_source_codes = []
	
	        for root, _, files in os.walk(self.directory_path):
	            for file in files:
	                if file.endswith(".py"):
	                    file_path = os.path.join(root, file)
	                    self._process_file(file_path)
	
	        function_source_codes = self.visitor.function_source_codes
	        function_nodes = self.visitor.function_nodes
	        class_source_codes = self.visitor.class_source_codes
	        class_nodes = self.visitor.class_nodes
	
	        return function_source_codes, class_source_codes, function_nodes, class_nodes
	
	    def clone_repo(self, repo_url):
	        repo_name = repo_url.split("/")[-1].replace(".git", "")
	        target_directory = os.path.join(self.directory_path, repo_name)
	
	        if os.path.exists(target_directory):
	            shutil.rmtree(target_directory)
	
	        subprocess.run(["git", "clone", repo_url, target_directory])
	
	        return target_directory
	

.. code-block:: python

	
	
	class GitHubUserProcessor:
	    def __init__(
	        self, username=None, repo_name=None, visitor=FunctionAndClassVisitor()
	    ):
	        self.username = username
	        self.repo_name = repo_name
	        self.github = Github()
	        self.directory_processor = None
	        self.function_source_codes = []
	        self.class_source_codes = []
	        self.visitor = visitor
	
	    def get_public_repos(self):
	        user = self.github.get_user(self.username)
	        return user.get_repos()
	
	    def process_repos(self, base_directory):
	        self.directory_processor = DirectoryProcessor(base_directory, self.visitor)
	        for repo in self.get_public_repos():
	            if not repo.private:
	                print(f"Processing repo: {repo.name}")
	                repo_path = self.directory_processor.clone_repo(repo.clone_url)
	                (
	                    function_source_codes,
	                    class_source_codes,
	                ) = self.directory_processor.process_directory()
	                self.function_source_codes.extend(function_source_codes)
	                self.class_source_codes.extend(class_source_codes)
	                shutil.rmtree(repo_path)
	
	        return self.directory_processor
	

.. code-block:: python

	
	
	class GitHubRepoProcessor:
	    def __init__(
	        self, username=None, repo_name=None, visitor=FunctionAndClassVisitor()
	    ):
	        self.username = username
	        self.repo_name = repo_name
	        self.github = Github()
	        self.directory_processor = None
	        self.function_source_codes = []
	        self.function_nodes = []
	        self.class_source_codes = []
	        self.class_nodes = []
	        self.visitor = visitor
	
	    def get_repo(self, repo_name):
	        user = self.github.get_user(self.username)
	        return user.get_repo(repo_name)
	
	    def process_repo(self, base_directory):
	        self.directory_processor = DirectoryProcessor(base_directory, self.visitor)
	        repo = self.get_repo(self.repo_name)
	        print(f"Processing repo: {self.repo_name}")
	        repo_path = self.directory_processor.clone_repo(repo.clone_url)
	        (
	            function_source_codes,
	            class_source_codes,
	            function_nodes,
	            class_nodes,
	        ) = self.directory_processor.process_directory()
	        self.function_source_codes.extend(function_source_codes)
	        self.function_nodes.extend(function_nodes)
	        self.class_source_codes.extend(class_source_codes)
	        self.class_nodes.extend(class_nodes)
	        shutil.rmtree(repo_path)
	        return self.directory_processor
	
	    def get_values(self):
	        # concatenate the function and class source codes
	        self.function_source_codes.extend(self.class_source_codes)
	        self.function_nodes.extend(self.class_nodes)
	        return self.function_source_codes, self.function_nodes
	

.. automodule:: git_processor
   :members:
