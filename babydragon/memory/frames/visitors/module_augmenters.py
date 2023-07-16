import libcst as cst
import libcst.matchers as matchers

class PassInserter(cst.CSTTransformer):
    def __init__(self):
        super().__init__()
        self.inside_class = False

    def visit_ClassDef(self, node: cst.ClassDef):
        self.inside_class = True
        return super().visit_ClassDef(node)

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
        self.inside_class = False
        return updated_node

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        body = updated_node.body
        body_elements = body.body

        # Check if the first element is a docstring

        #if body_elements and matchers.matches(body_elements[0], matchers.Expr(matchers.SimpleString() | matchers.ConcatenatedString())):
            #docstring = body_elements[0]
        docstrings = []
        for element in body_elements:
            if matchers.matches(element, matchers.Expr(matchers.SimpleString() | matchers.ConcatenatedString())):
                docstrings.append(element)

        # Prepare new body
        new_body = [cst.SimpleStatementLine(body=(cst.Pass(),))]
        if docstrings[0] is not None:
            new_body.insert(0, docstrings[0])

        return updated_node.with_changes(
            body=cst.IndentedBlock(body=tuple(new_body))
        )

def generate_skeleton(code: str) -> str:
    """
    Generate a skeleton for the given code, replacing function and class bodies with a pass statement.
    """
    # Parse the code into a CST
    module = cst.parse_module(code)

    # Apply the transformer
    transformer = PassInserter()
    transformed_module = module.visit(transformer)

    # Convert the transformed CST back to code
    skeleton_code = transformed_module.code

    return skeleton_code

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