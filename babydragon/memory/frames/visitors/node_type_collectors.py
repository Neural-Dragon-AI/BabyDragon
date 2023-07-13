import libcst as cst

class DocstringCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.docstrings = []

    def visit_Module(self, node: cst.Module) -> bool:
        if node.body and isinstance(node.body[0].body, cst.SimpleStatementLine):
            for stmt in node.body[0].body.body:
                if isinstance(stmt, cst.Expr) and isinstance(stmt.value, cst.SimpleString):
                    self.docstrings.append(stmt.value.value)
        return True

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        docstring = node.get_docstring()
        if docstring is not None:
            self.docstrings.append(docstring)
        return True

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        docstring = node.get_docstring()
        if docstring is not None:
            self.docstrings.append(docstring)
        return True

    def collect(self):
        self.module.visit(self)
        return self.docstrings

class FunctionCallCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.function_calls = []

    def visit_Call(self, node: cst.Call) -> bool:
        if isinstance(node.func, cst.Name):
            # Add the function name to the list
            self.function_calls.append(node.func.value)
        return True

    def collect(self):
        self.module.visit(self)
        return self.function_calls

class ArgumentTypeCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.argument_types = []

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        # Collect argument types for functions
        arg_types = []
        for param in node.params.params:
            if isinstance(param.annotation, cst.Annotation):
                arg_types.append(param.annotation.annotation.value)
            else:
                arg_types.append(None)

        self.argument_types.append(arg_types)
        return True

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        # Collect argument types for methods
        for stmt in node.body.body:
            if isinstance(stmt, cst.FunctionDef):
                arg_types = []
                for param in stmt.params.params:
                    if isinstance(param.annotation, cst.Annotation):
                        arg_types.append(param.annotation.annotation.value)
                    else:
                        arg_types.append(None)

                self.argument_types.append(arg_types)

        return True

    def collect(self):
        self.module.visit(self)
        return self.argument_types

class ImportCollector(cst.CSTVisitor):
    def __init__(self, filename: str):
        with open(filename, "r") as file:
            self.module = cst.parse_module(file.read())
        self.imports = []

    def visit_Import(self, node: cst.Import) -> bool:
        for name in node.names:
            self.imports.append(cst.Module([node]))
        return True

    def visit_ImportFrom(self, node: cst.ImportFrom) -> bool:
        module = node.module.value if node.module else ""
        for name in node.names:
            self.imports.append(cst.Module([node]))
        return True

    def collect(self):
        self.module.visit(self)
        import_code = [import_statement.code for import_statement in self.imports]
        return import_code

class IfStatementCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.if_statements = []

    def visit_If(self, node: cst.If) -> bool:
        self.if_statements.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.if_statements

class BaseCompoundStatementCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.compound_statements = []

    def visit_BaseCompoundStatement(self, node: cst.BaseCompoundStatement) -> bool:
        self.compound_statements.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.compound_statements
    
class ForLoopCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.for_loops = []

    def visit_For(self, node: cst.For) -> bool:
        self.for_loops.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.for_loops

class WhileLoopCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.while_loops = []

    def visit_While(self, node: cst.While) -> bool:
        self.while_loops.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.while_loops
    
class TryExceptCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.try_excepts = []

    def visit_Try(self, node: cst.Try) -> bool:
        self.try_excepts.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.try_excepts

class WithCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.withs = []

    def visit_With(self, node: cst.With) -> bool:
        self.withs.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.withs

class VariableDeclarationCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.variable_declarations = []

    def visit_Assign(self, node: cst.Assign) -> bool:
        self.variable_declarations.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.variable_declarations

class ListComprehensionCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.list_comprehensions = []

    def visit_ListComp(self, node: cst.ListComp) -> bool:
        self.list_comprehensions.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.list_comprehensions

class DictComprehensionCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.dict_comprehensions = []

    def visit_DictComp(self, node: cst.DictComp) -> bool:
        self.dict_comprehensions.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.dict_comprehensions


# Set Comprehension Collector
class SetComprehensionCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.set_comprehensions = []

    def visit_SetComp(self, node: cst.SetComp) -> bool:
        self.set_comprehensions.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.set_comprehensions

# Generator Expression Collector
class GeneratorExpressionCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.generator_expressions = []

    def visit_GeneratorExp(self, node: cst.GeneratorExp) -> bool:
        self.generator_expressions.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.generator_expressions

# Yield Statement Collector
class YieldCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.yields = []

    def visit_Yield(self, node: cst.Yield) -> bool:
        self.yields.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.yields


# Return Statement Collector
class ReturnCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.returns = []

    def visit_Return(self, node: cst.Return) -> bool:
        self.returns.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.returns

# Raise Statement Collector
class RaiseCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.raises = []

    def visit_Raise(self, node: cst.Raise) -> bool:
        self.raises.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.raises

# Assert Statement Collector
class AssertCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.asserts = []

    def visit_Assert(self, node: cst.Assert) -> bool:
        self.asserts.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.asserts

# Break Statement Collector
class BreakCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.breaks = []

    def visit_Break(self, node: cst.Break) -> bool:
        self.breaks.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.breaks

# Continue Statement Collector
class ContinueCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.continues = []

    def visit_Continue(self, node: cst.Continue) -> bool:
        self.continues.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.continues

# Pass Statement Collector
class PassCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.passes = []

    def visit_Pass(self, node: cst.Pass) -> bool:
        self.passes.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.passes



# With Statement Collector
class WithStatementCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.with_statements = []

    def visit_With(self, node: cst.With) -> bool:
        self.with_statements.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.with_statements

# Try Statement Collector
class TryStatementCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.try_statements = []

    def visit_Try(self, node: cst.Try) -> bool:
        self.try_statements.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.try_statements

# Except Clause Collector
class ExceptClauseCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.except_clauses = []

    def visit_ExceptHandler(self, node: cst.ExceptHandler) -> bool:
        self.except_clauses.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.except_clauses

# Lambda Function Collector
class LambdaFunctionCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.lambda_functions = []

    def visit_Lambda(self, node: cst.Lambda) -> bool:
        self.lambda_functions.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.lambda_functions

# Global Statement Collector
class GlobalStatementCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.global_statements = []

    def visit_Global(self, node: cst.Global) -> bool:
        self.global_statements.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.global_statements

# Nonlocal Statement Collector
class NonlocalStatementCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.nonlocal_statements = []

    def visit_Nonlocal(self, node: cst.Nonlocal) -> bool:
        self.nonlocal_statements.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.nonlocal_statements