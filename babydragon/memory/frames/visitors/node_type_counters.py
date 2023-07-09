import libcst as cst
# Function Call Counter
class FunctionCallCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.function_call_count = 0

    def visit_Call(self, node: cst.Call) -> bool:
        self.function_call_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.function_call_count

# Argument Type Counter
class ArgumentTypeCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.argument_type_count = 0

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        self.argument_type_count += len(node.params.params)
        return True

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        for stmt in node.body.body:
            if isinstance(stmt, cst.FunctionDef):
                self.argument_type_count += len(stmt.params.params)
        return True

    def collect(self):
        self.module.visit(self)
        return self.argument_type_count

# Import Counter
class ImportCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.import_count = 0

    def visit_Import(self, node: cst.Import) -> bool:
        self.import_count += 1
        return True

    def visit_ImportFrom(self, node: cst.ImportFrom) -> bool:
        self.import_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.import_count

# If Statement Counter
class IfStatementCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.if_statement_count = 0

    def visit_If(self, node: cst.If) -> bool:
        self.if_statement_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.if_statement_count

# Base Compound Statement Counter
class BaseCompoundStatementCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.compound_statement_count = 0

    def visit_BaseCompoundStatement(self, node: cst.BaseCompoundStatement) -> bool:
        self.compound_statement_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.compound_statement_count

# For Loop Counter
class ForLoopCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.for_loop_count = 0

    def visit_For(self, node: cst.For) -> bool:
        self.for_loop_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.for_loop_count

# While Loop Counter
class WhileLoopCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.while_loop_count = 0

    def visit_While(self, node: cst.While) -> bool:
        self.while_loop_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.while_loop_count

# Try Except Counter
class TryExceptCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.try_except_count = 0

    def visit_Try(self, node: cst.Try) -> bool:
        self.try_except_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.try_except_count

# With Statement Counter
class WithStatementCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.with_statement_count = 0

    def visit_With(self, node: cst.With) -> bool:
        self.with_statement_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.with_statement_count

# Lambda Function Counter
class LambdaFunctionCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.lambda_function_count = 0

    def visit_Lambda(self, node: cst.Lambda) -> bool:
        self.lambda_function_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.lambda_function_count

# Global Statement Counter
class GlobalStatementCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.global_statement_count = 0

    def visit_Global(self, node: cst.Global) -> bool:
        self.global_statement_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self

# Nonlocal Statement Counter
class NonlocalStatementCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.nonlocal_statement_count = 0

    def visit_Nonlocal(self, node: cst.Nonlocal) -> bool:
        self.nonlocal_statement_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.nonlocal_statement_count