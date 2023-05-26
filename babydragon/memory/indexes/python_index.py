from typing import Optional

import tiktoken
import os
from babydragon.memory.indexes.memory_index import MemoryIndex
from babydragon.processors.parsers.python_parser import PythonParser


class PythonIndex(MemoryIndex, PythonParser):
    def __init__(
        self,
        directory_path: str,
        name: str = "python_index",
        save_path: Optional[str] = None,
        load: bool = False,
        minify_code: bool = False,
        remove_docstrings: bool = False,
        tokenizer: Optional[tiktoken.Encoding] = None,
        max_workers: int = 1,
        backup: bool = False,
        filter: str = "class_function"
    ):

        PythonParser.__init__(
            self,
            directory_path=directory_path,
            minify_code=minify_code,
            remove_docstrings=remove_docstrings,
        )
        #check if load folder exists
        if save_path is None:
            save_path = "storage"
        load_directory = os.path.join(save_path, name)
        loadcheck = not load or not os.path.exists(load_directory)
        if load and not os.path.exists(load_directory):
            print("No python-index found even if load=True, indexing from scratch")
        if loadcheck:
            # Extract functions and classes source code
            function_source_codes, class_source_codes, _, _ = self.process_directory()
            print(
                "Indexing {} functions and {} classes".format(
                    len(function_source_codes), len(class_source_codes)
                )
            )
            # Concatenate function and class source code and index them
            if filter == "function":
                codes = function_source_codes
            elif filter == "class":
                codes = class_source_codes
            elif filter == "class_function":
                codes = function_source_codes + class_source_codes
            load = False
            self.function_source_codes = function_source_codes
            self.class_source_codes = class_source_codes

         # Initialize the MemoryIndex
        MemoryIndex.__init__(
            self,
            name=name,
            values=codes if loadcheck else None,
            save_path=save_path,
            load=load,
            tokenizer=tokenizer,
            max_workers=max_workers,
            backup=backup,
        )
        self.markdown = "python"