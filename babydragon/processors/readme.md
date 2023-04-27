# Processors submodule for BabyDragon library

The processors submodule provides classes to work with files, directories, and Python code. It also provides functionality for processing GitHub repositories.

## OsProcessor

The base class that provides various utility methods to work with directories and files:

- `get_all_files`
- `get_files_with_extension`
- `get_file_extension`
- `get_subdirectories`
- `create_directory`
- `delete_directory`
- `copy_file`
- `move_file`

## PythonMinifier

A class responsible for minifying Python source code using the `python_minifier` package:

- `minify`
- `get_minified_code`
- `minify_code` (staticmethod)

## PythonDocstringExtractor

A class providing a static method to extract the docstring from a given function definition:

- `extract_docstring`

## FunctionAndClassVisitor

A class that inherits from `cst.CSTVisitor` and collects information about function and class definitions in a Python source code file:

- `visit_FunctionDef`
- `visit_ClassDef`

## PythonParser

A class that inherits from `OsProcessor` and is used to process a directory of Python files:

- `remove_docstring`
- `_process_file`
- `process_file`
- `process_directory`

```python
# Example usage of PythonParser
directory_path = "/path/to/directory"
parser = PythonParser(directory_path, minify_code=True, remove_docstrings=True)
function_source_codes, class_source_codes, function_nodes, class_nodes = parser.process_directory()
```
## GithubProcessor

A class that inherits from `OsProcessor` and provides functionality for processing GitHub repositories:

- `get_public_repos`
- `clone_repo`
- `process_repo`
- `process_repos`
- `get_repo`
- `process_single_repo`
- `get_issues`
- `parse_issues`
- `get_commits`
- `parse_commits`

```python
# Example usage of GithubProcessor
base_directory = "/path/to/base_directory"
username = "github_username"
repo_name = "repository_name"
processor = GithubProcessor(base_directory, username, repo_name)
processor.process_single_repo()
```