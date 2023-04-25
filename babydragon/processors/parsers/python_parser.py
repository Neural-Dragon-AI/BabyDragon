import os
import subprocess
import libcst as cst
from typing import List, Tuple, Union, Optional
from babydragon.processors.os_processor import OsProcessor
from python_minifier import minify


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
                    if isinstance(expr, cst.Expr) and isinstance(expr.value, cst.SimpleString):
                        docstring = expr.value.value.strip('"').strip("'")
                        break
            if docstring is not None:
                break

        if docstring is not None:
            return docstring.strip()
        else:
            function_name = function_def.name.value
            return f"No docstring provided for function '{function_name}'. Please add a docstring to describe this function."

class FunctionAndClassVisitor(cst.CSTVisitor):
    def __init__(self):
        self.function_source_codes = []
        self.function_nodes = []
        self.class_source_codes = []
        self.class_nodes = []

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        """ This method is called for every FunctionDef node in the tree.
        and it does the following:
        1. Gets the source code for the node
        2. Adds the node to the list of function nodes
        3. Adds the source code to the list of function source codes
        """
        function_source_code = cst.Module([]).code_for_node(node)
        self.function_nodes.append(node)
        self.function_source_codes.append(function_source_code)

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        """ This method is called for every ClassDef node in the tree.
        and it does the following:
        1. Gets the source code for the node
        2. Adds the node to the list of class nodes
        3. Adds the source code to the list of class source codes
        """
        class_source_code = cst.Module([]).code_for_node(node)
        self.class_nodes.append(node)
        self.class_source_codes.append(class_source_code)


class PythonParser(OsProcessor):
    def __init__(self, directory_path: str, visitor: Optional[FunctionAndClassVisitor] = None, minify_code: bool = False, remove_docstrings: bool = False):
        super().__init__(directory_path)
        self.visitor = visitor if visitor else FunctionAndClassVisitor()
        self.minify_code = minify_code
        self.remove_docstrings = remove_docstrings

    
    def remove_docstring(self, tree: cst.Module) -> str:
        """Removes docstrings from the given code and returns the code without docstrings."""

        # Remove docstrings using a transformer
        class DocstringRemover(cst.CSTTransformer):
            def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
                docstring = PythonDocstringExtractor.extract_docstring(original_node)
                if docstring.startswith("No docstring"):
                    return updated_node

                return updated_node.with_changes(body=updated_node.body.with_changes(body=[stmt for stmt in updated_node.body.body if not (isinstance(stmt, cst.SimpleStatementLine) and any(isinstance(expr, cst.Expr) and isinstance(expr.value, cst.SimpleString) for expr in stmt.body))]))

        tree = tree.visit(DocstringRemover())
        return tree.code

    def _process_file(self, file_path: str):
        """ This method is called for every file in the directory.
        It does the following:
        1. Reads the file
        2. Parses the file
        3. Visits the file with the visitor
        """
        with open(file_path, "r", encoding='utf-8') as file:
            source_code = file.read()

        try:
            tree = cst.parse_module(source_code)
        except cst.ParserSyntaxError:
            print(f"Skipping file {file_path}: Failed to parse syntax")
            return

        tree.visit(self.visitor)

        # Remove docstrings if specified
        if self.remove_docstrings:
            source_code = self.remove_docstring(source_code, tree)

        # Minify the code if specified
        if self.minify_code:
            minifier = PythonMinifier(source_code)
            source_code = minifier.get_minified_code()

        # Add the processed code to the corresponding list in the visitor
        self.visitor.function_source_codes.append(source_code)

    def process_file(self, file_path: str):
        """ This method is called for every file in the directory.
        It does the following:
        1. Runs flake8 on the file
        if flake8 returns a non-zero exit code, it means the file has a syntax error
        2. Reads the file
        3. Parses the file
        4. Visits the file with the visitor

        """
        result = subprocess.run(["flake8", "--select=E999", file_path], capture_output=True)

        if result.returncode != 0:
            print(f"Skipping file with syntax error: {file_path}")
            print(result.stderr.decode("utf-8"))
            return

        with open(file_path, "r", encoding='utf-8') as f:
            source_code = f.read()

        try:
            tree = cst.parse_module(source_code)
            tree.visit(self.visitor)
        except cst.ParserSyntaxError as e:
            print(f"Syntax error: {e}")
            print(f"Skipping file with syntax error: {file_path}")

    def process_directory(self) -> Tuple[List[str], List[str], List[cst.FunctionDef], List[cst.ClassDef]]:
        """ This method is called for every directory.
        It does the following:
        1. Gets all the python files in the directory
        2. Processes each file
        3. Returns the list of function source codes, class source codes, function nodes, and class nodes
        """
        function_source_codes = []
        class_source_codes = []

        python_files = self.get_files_with_extension('.py')

        for file_path in python_files:
            self._process_file(file_path)

        function_source_codes = self.visitor.function_source_codes
        function_nodes = self.visitor.function_nodes
        class_source_codes = self.visitor.class_source_codes
        class_nodes = self.visitor.class_nodes

        return function_source_codes, class_source_codes, function_nodes, class_nodes
