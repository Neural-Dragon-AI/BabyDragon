from babydragon.processors.parsers.python_parser import PythonParser
from typing import Dict, List, Tuple
import os
import glob
import codecs

def traverse_and_collect_rtd(directory):
    '''
    This function traverses a directory and collects content from .rst files.
    The content is stored in a list of tuples, with filename as 0th element and content as 1st.

    Parameters:
    directory (str): The directory to traverse

    Returns:
    list: A list of tuples containing filename and file content
    '''

    # Placeholder for the list of tuples
    data = []

    # Traversing the directory recursively
    for filename in glob.glob(os.path.join(directory, '**', '*.rst'), recursive=True):

        # Open file
        with codecs.open(filename, 'r', encoding='utf-8', errors='ignore') as file:

            # Read content and add to list
            content = file.read()
            data.append((filename, content))

    return data

def extract_values_python(
    directory_path: str,
    minify_code: bool = False,
    remove_docstrings: bool = False,
    resolution: str = "function"
) -> Tuple[List[str], List[Dict[str, str]]]:
    values = []
    context = []

    parser = PythonParser(
        directory_path=directory_path,
        minify_code=minify_code,
        remove_docstrings=remove_docstrings
    )

    results_dict = parser.process_directory()

    if resolution == "function":
        source_codes = results_dict['function_source_codes']
        nodes = results_dict['function_nodes']
    elif resolution == "class":
        source_codes = results_dict['class_source_codes']
        nodes = results_dict['class_nodes']
    elif resolution == "both":
        source_codes = results_dict['full_source']
        nodes = results_dict['full_nodes']
    else:
        raise ValueError(f"Invalid resolution: {resolution}")
    if resolution in ['function', 'class']:
        for source_code, node in zip(source_codes, nodes):
            values.append(source_code)
            context.append({"libcst tree": str(node)})
    elif resolution == "both":
        for source_code, node, filename in zip(source_codes, nodes, results_dict['file_map']):
            values.append(source_code)
            context.append({"libcst tree": str(node), "filename": filename})
    else:
        raise ValueError(f"Invalid resolution: {resolution}")
    return values, context


