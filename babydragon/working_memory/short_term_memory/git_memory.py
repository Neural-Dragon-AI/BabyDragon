from typing import List, Optional, Union

import libcst as cst
from python_minifier import minify

from babydragon.memory.indexes.memory_index import MemoryIndex
from babydragon.working_memory.parsers.git_processor import GitHubRepoProcessor


class PythonMinifier:
    def __init__(self, code: str = None):

        self.code = code
        self.output_code = None

    def minify(self):
        if self.code:
            self.output_code = self.minify_code(self.code)

    def get_minified_code(self):
        if not self.output_code:
            self.minify()
        return self.output_code

    @staticmethod
    def minify_code(code: str) -> str:
        return minify(code)


class PythonDocstringExtractor:
    @staticmethod
    def extract_docstring(function_def: cst.FunctionDef) -> str:
        docstring = None

        for stmt in function_def.body.body:
            if isinstance(stmt, cst.SimpleStatementLine):
                for expr in stmt.body:
                    if isinstance(expr, cst.Expr) and isinstance(
                        expr.value, cst.SimpleString
                    ):
                        docstring = expr.value.value.strip('"').strip("'")
                        break
            if docstring is not None:
                break

        if docstring is not None:
            return docstring.strip()
        else:
            function_name = function_def.name.value
            return f"No docstring provided for function '{function_name}'. Please add a docstring to describe this function."


class GitMemory(MemoryIndex):
    def __init__(self, username, repo_name):
        super().__init__()
        self.username = username
        self.repo_name = repo_name
        self.parser = GitHubRepoProcessor(username, repo_name)
        self.minifier = PythonMinifier()
        self.docstring_extractor = PythonDocstringExtractor()
        self.directory_parser = None
        self.min_code_index = None
        self.doc_string_index = None
        self.libcst_node_index = None

    def create_code_index(self, base_directory):
        self.directory_parser = self.parser.process_repo(base_directory)
        code_values, code_nodes = self.parser.get_values()
        self.code_index = self.init_index(values=code_values)
        self.code_index.save()

    def create_indexes(self, base_directory):
        self.directory_parser = self.parser.process_repo(base_directory)
        code_values, code_nodes = self.parser.get_values()
        self.code_index = self.init_index(values=code_values)

        min_code_values = []
        doc_string_values = []
        for code_value, code_node in zip(code_values, code_nodes):
            minifier = PythonMinifier(code=code_value)
            min_code = minifier.get_minified_code()
            doc_string = self.docstring_extractor.extract_docstring(code_node)
            min_code_values.append(min_code)
            doc_string_values.append(doc_string)
        self.doc_string_index = self.init_index(values=doc_string_values)
        self.min_code_index = self.init_index(values=min_code_values)
