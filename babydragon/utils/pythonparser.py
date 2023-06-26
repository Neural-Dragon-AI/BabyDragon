from babydragon.processors.parsers.python_parser import PythonParser
from typing import Dict, List, Tuple

def extract_values_and_embeddings_python(
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
            context.append({"libcst tree": node})
    elif resolution == "both":
        for source_code, node, filename in zip(source_codes, nodes, results_dict['file_map']):
            values.append(source_code)
            context.append({"libcst tree": node, "filename": filename})
    else:
        raise ValueError(f"Invalid resolution: {resolution}")
    return values, context


