from typing import Optional
from babydragon.memory.indexes.memory_index import MemoryIndex
from babydragon.processors.parsers.python_parser import PythonParser
import tiktoken

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
    ):
        # Initialize the MemoryIndex
        MemoryIndex.__init__(
            self,
            name=name,
            save_path=save_path,
            load=load,
            tokenizer=tokenizer,
        )
        # Initialize the PythonParser
        PythonParser.__init__(
            self,
            directory_path=directory_path,
            minify_code=minify_code,
            remove_docstrings=remove_docstrings,
        )

        if not load:
            # Extract functions and classes source code
            function_source_codes, class_source_codes, _, _ = self.process_directory()
            print("Indexing {} functions and {} classes".format(len(function_source_codes), len(class_source_codes)))
            # Concatenate function and class source code and index them
            codes = function_source_codes + class_source_codes
            for code in codes:
                self.add_to_index(code)


            self.save()