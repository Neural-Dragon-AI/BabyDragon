import libcst as cst
# Function Call Counter

NODETYPE_COUNTERS = [
    'FunctionCallCounter',
    'ArgumentTypeCounter',
    'ImportCounter',
    'IfStatementCounter',
    'BaseCompoundStatementCounter',
    'ForLoopCounter',
    'WhileLoopCounter',
    'TryExceptCounter',
    'WithStatementCounter',
    'LambdaFunctionCounter',
    'GlobalStatementCounter',
    'NonlocalStatementCounter',
    'ListComprehensionCounter',
    'DictComprehensionCounter',
    'SetComprehensionCounter',
    'GeneratorExpressionCounter',
    'AwaitCounter',
    'ReturnCounter',
    'BreakCounter',
    'ContinueCounter',
    'RaiseCounter',
    'AssertCounter',
    'PassCounter'
]

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
        return self.global_statement_count

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

class ListComprehensionCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.list_comprehension_count = 0

    def visit_ListComp(self, node: cst.ListComp) -> bool:
        self.list_comprehension_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.list_comprehension_count

class DictComprehensionCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.dict_comprehension_count = 0

    def visit_DictComp(self, node: cst.DictComp) -> bool:
        self.dict_comprehension_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.dict_comprehension_count

class SetComprehensionCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.set_comprehension_count = 0

    def visit_SetComp(self, node: cst.SetComp) -> bool:
        self.set_comprehension_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.set_comprehension_count

class GeneratorExpressionCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.generator_expression_count = 0

    def visit_GeneratorExp(self, node: cst.GeneratorExp) -> bool:
        self.generator_expression_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.generator_expression_count

class YieldCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.yield_count = 0

    def visit_Yield(self, node: cst.Yield) -> bool:
        self.yield_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.yield_count

class AwaitCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.await_count = 0

    def visit_Await(self, node: cst.Await) -> bool:
        self.await_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.await_count

class ReturnCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.return_count = 0

    def visit_Return(self, node: cst.Return) -> bool:
        self.return_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.return_count

class BreakCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.break_count = 0

    def visit_Break(self, node: cst.Break) -> bool:
        self.break_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.break_count

class ContinueCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.continue_count = 0

    def visit_Continue(self, node: cst.Continue) -> bool:
        self.continue_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.continue_count

class RaiseCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.raise_count = 0

    def visit_Raise(self, node: cst.Raise) -> bool:
        self.raise_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.raise_count

class AssertCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.assert_count = 0

    def visit_Assert(self, node: cst.Assert) -> bool:
        self.assert_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.assert_count


class PassCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.pass_count = 0

    def visit_Pass(self, node: cst.Pass) -> bool:
        self.pass_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.pass_count

