
from babydragon.memory.indexes.memory_index import MemoryIndex
from babydragon.working_memory.parsers.git_processor import GitHubRepoProcessor
import re

class PythonMinifier:

    @staticmethod
    def remove_comments(line):
        return re.sub(r"(\s*#.*)", "", line)

    @staticmethod
    def remove_empty_lines(lines):
        return [line for line in lines if line.strip()]

    @staticmethod
    def remove_extra_spaces(lines):
        return [re.sub(r"(\s+)", " ", line) for line in lines]

    @staticmethod
    def minify(code):
        lines = code.splitlines()
        lines = [PythonMinifier.remove_comments(line) for line in lines]
        lines = PythonMinifier.remove_extra_spaces(lines)
        lines = PythonMinifier.remove_empty_lines(lines)
        return "\n".join(lines)

class GitMemory(MemoryIndex):
    def __init__(self, username, repo_name):
        super().__init__()
        self.username = username
        self.repo_name = repo_name
        self.parser = GitHubRepoProcessor(username, repo_name)
        self.minifier = PythonMinifier()
        self.directory_parser = None
        self.code_index = None
        self.min_code_index = None
        self.summary_index = None

    def create_indexes(self, base_directory):
        self.directory_parser = self.parser.process_repo(base_directory)
        code_values = self.directory_parser.get_values()
        self.code_index = self.init_index(code_values)
