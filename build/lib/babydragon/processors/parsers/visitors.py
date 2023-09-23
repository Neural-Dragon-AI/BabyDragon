from typing import Dict, List, Optional, Tuple

import libcst as cst
import libcst.matchers as m
from libcst.metadata import PositionProvider


# A custom visitor to find function calls and their arguments
class FunctionCallFinder(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def visit_Call(self, node: cst.Call) -> None:
        function_name = None
        if isinstance(node.func, cst.Name):
            function_name = node.func.value

        if function_name:
            pos = self.get_metadata(PositionProvider, node).start
            print(
                f"Function '{function_name}' called at line {pos.line}, column {pos.column} with arguments:"
            )

            for arg in node.args:
                arg_start_pos = self.get_metadata(PositionProvider, arg).start
                arg_value = arg.value
                if isinstance(arg_value, cst.SimpleString):
                    arg_value = arg_value.evaluated_value
                print(
                    f"- Argument at line {arg_start_pos.line}, column {arg_start_pos.column}: {arg_value}"
                )


class MultiplicationCounterVisitor(cst.CSTVisitor):
    def __init__(self):
        self.count = 0
        self.functions_with_operation_dict = {}

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        self.current_function = node
        self.functions_with_operation_dict[node.name] = []

    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
        self.current_function = None

    def visit_BinaryOperation(self, node: cst.BinaryOperation) -> None:
        if isinstance(node.operator, cst.Multiply) or isinstance(
            node.operator, cst.BitAnd
        ):
            self.count += 1
            if self.current_function:
                self.functions_with_operation_dict[self.current_function.name].append(
                    cst.Module([]).code_for_node(node)
                )

    def visit_Call(self, node: cst.Call) -> None:
        if m.matches(node, m.Call(func=m.Attribute(attr=m.Name("dot")))) or m.matches(
            node, m.Call(func=m.Name("dot"), args=[m.Arg(), m.Arg()])
        ):
            self.count += 1
            if self.current_function:
                self.functions_with_operation_dict[self.current_function.name].append(
                    cst.Module([]).code_for_node(node)
                )


class FunctionAndClassVisitor(cst.CSTVisitor):
    def __init__(self):
        self.function_source_codes = []
        self.function_nodes = []
        self.class_source_codes = []
        self.class_nodes = []

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        function_source_code = cst.Module([]).code_for_node(node)
        # add in place summary and code mod
        self.function_nodes.append(node)
        self.function_source_codes.append(function_source_code)

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        class_source_code = cst.Module([]).code_for_node(node)
        # add in place summary and code mod
        self.class_nodes.append(node)
        self.class_source_codes.append(class_source_code)


class TypingCollector(cst.CSTVisitor):
    def __init__(self):
        # stack for storing the canonical name of the current function
        self.stack: List[Tuple[str, ...]] = []
        # store the annotations
        self.annotations: Dict[
            Tuple[str, ...],  # key: tuple of canonical class/function name
            Tuple[cst.Parameters, Optional[cst.Annotation]],  # value: (params, returns)
        ] = {}

    def visit_ClassDef(self, node: cst.ClassDef) -> Optional[bool]:
        self.stack.append(node.name.value)

    def leave_ClassDef(self, node: cst.ClassDef) -> None:
        self.stack.pop()

    def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
        self.stack.append(node.name.value)
        self.annotations[tuple(self.stack)] = (node.params, node.returns)
        return False  # pyi files don't support inner functions, return False to stop the traversal.

    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
        self.stack.pop()
