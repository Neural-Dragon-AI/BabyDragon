import libcst as cst

class CodeReplacerVisitor(cst.CSTTransformer):
    def __init__(self, filename_column: str, original_code_column: str, replacing_code_column: str):
        self.filename_column = filename_column
        self.original_code_column = original_code_column
        self.replacing_code_column = replacing_code_column

    def visit_Module(self, node: cst.Module) -> cst.Module:
        # Get the filename from the node metadata
        filename = node.metadata.get(self.filename_column)
        if filename is None:
            return node

        # Load the content of the file
        with open(filename, "r") as f:
            file_content = f.read()

        # Get the original code and replacing code from the node metadata
        original_code = node.metadata.get(self.original_code_column)
        replacing_code = node.metadata.get(self.replacing_code_column)

        if original_code is None or replacing_code is None:
            return node

        # Replace the original code with the replacing code in the file content
        modified_content = file_content.replace(original_code, replacing_code)

        # Save the modified content back to the file
        with open(filename, "w") as f:
            f.write(modified_content)

        # Parse the modified content to update the node
        return cst.parse_module(modified_content)

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        # Copy the metadata from the original node to the updated node
        updated_node.metadata = original_node.metadata
        return updated_node