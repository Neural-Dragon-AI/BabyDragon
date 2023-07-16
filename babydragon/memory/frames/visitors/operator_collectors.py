import libcst as cst

# Unary Operators
class UnaryOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.unary_operators = []

    def visit_UnaryOperation(self, node: cst.UnaryOperation) -> bool:
        if isinstance(node.operator, cst.BitInvert) or isinstance(node.operator, cst.Minus) or \
                isinstance(node.operator, cst.Not) or isinstance(node.operator, cst.Plus):
            self.unary_operators.append(node.operator.__class__.__name__)
        return True

    def collect(self):
        self.module.visit(self)
        return self.unary_operators

# Boolean Operators
class BooleanOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.boolean_operators = []

    def visit_BooleanOperation(self, node: cst.BooleanOperation) -> bool:
        if isinstance(node.operator, cst.And) or isinstance(node.operator, cst.Or):
            self.boolean_operators.append(node.operator.__class__.__name__)
        return True

    def collect(self):
        self.module.visit(self)
        return self.boolean_operators

# Binary Operators
class BinaryOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.binary_operators = []

    def visit_BinaryOperation(self, node: cst.BinaryOperation) -> bool:
        if isinstance(node.operator, cst.Add) or isinstance(node.operator, cst.BitAnd) or \
                isinstance(node.operator, cst.BitOr) or isinstance(node.operator, cst.BitXor) or \
                isinstance(node.operator, cst.Divide) or isinstance(node.operator, cst.FloorDivide) or \
                isinstance(node.operator, cst.LeftShift) or isinstance(node.operator, cst.MatrixMultiply) or \
                isinstance(node.operator, cst.Modulo) or isinstance(node.operator, cst.Multiply) or \
                isinstance(node.operator, cst.Power) or isinstance(node.operator, cst.RightShift) or \
                isinstance(node.operator, cst.Subtract):
            self.binary_operators.append(node.operator.__class__.__name__)
        return True

    def collect(self):
        self.module.visit(self)
        return self.binary_operators

# Comparison Operators
class ComparisonOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.comparison_operators = []

    def visit_Comparison(self, node: cst.Comparison) -> bool:
        for operator in node.operators:
            if isinstance(operator, cst.Equal) or isinstance(operator, cst.GreaterThan) or \
                    isinstance(operator, cst.GreaterThanEqual) or isinstance(operator, cst.In) or \
                    isinstance(operator, cst.Is) or isinstance(operator, cst.LessThan) or \
                    isinstance(operator, cst.LessThanEqual) or isinstance(operator, cst.NotEqual) or \
                    isinstance(operator, cst.IsNot) or isinstance(operator, cst.NotIn):
                self.comparison_operators.append(operator.__class__.__name__)
        return True

    def collect(self):
        self.module.visit(self)
        return self.comparison_operators

# Augmented Assignment Operators
class AugmentedAssignmentOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.augmented_assignment_operators = []

    def visit_AugAssign(self, node: cst.AugAssign) -> bool:
        if isinstance(node.operator, cst.AddAssign) or isinstance(node.operator, cst.BitAndAssign) or \
                isinstance(node.operator, cst.BitOrAssign) or isinstance(node.operator, cst.BitXorAssign) or \
                isinstance(node.operator, cst.DivideAssign) or isinstance(node.operator, cst.FloorDivideAssign) or \
                isinstance(node.operator, cst.LeftShiftAssign) or isinstance(node.operator, cst.MatrixMultiplyAssign) or \
                isinstance(node.operator, cst.ModuloAssign) or isinstance(node.operator, cst.MultiplyAssign) or \
                isinstance(node.operator, cst.PowerAssign) or isinstance(node.operator, cst.RightShiftAssign) or \
                isinstance(node.operator, cst.SubtractAssign):
            self.augmented_assignment_operators.append(node.operator.__class__.__name__)
        return True

    def collect(self):
        self.module.visit(self)
        return self.augmented_assignment_operators

# Miscellaneous Operators
class MiscellaneousOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.miscellaneous_operators = []

    def visit_AssignEqual(self, node: cst.AssignEqual) -> bool:
        self.miscellaneous_operators.append(cst.Module([node]).code)
        return True

    def visit_Colon(self, node: cst.Colon) -> bool:
        self.miscellaneous_operators.append(cst.Module([node]).code)
        return True

    def visit_Comma(self, node: cst.Comma) -> bool:
        self.miscellaneous_operators.append(cst.Module([node]).code)
        return True

    def visit_Dot(self, node: cst.Dot) -> bool:
        self.miscellaneous_operators.append(cst.Module([node]).code)
        return True

    def visit_ImportStar(self, node: cst.ImportStar) -> bool:
        self.miscellaneous_operators.append(cst.Module([node]).code)
        return True

    def visit_Semicolon(self, node: cst.Semicolon) -> bool:
        self.miscellaneous_operators.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.miscellaneous_operators


# Unary Operators
class BitInvertOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.bit_invert_count = []

    def visit_BitInvert(self, node: cst.BitInvert) -> bool:
        self.bit_invert_count.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.bit_invert_count

class MinusOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.minus_count = []

    def visit_Minus(self, node: cst.Minus) -> bool:
        self.minus_count.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.minus_count

class NotOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.not_count = []

    def visit_Not(self, node: cst.Not) -> bool:
        self.not_count.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.not_count

class PlusOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.plus_count = []

    def visit_Plus(self, node: cst.Plus) -> bool:
        self.plus_count.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.plus_count


# Boolean Operators
class AndOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.and_count = []

    def visit_And(self, node: cst.And) -> bool:
        self.and_count.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.and_count

class OrOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.or_count = []

    def visit_Or(self, node: cst.Or) -> bool:
        self.or_count.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.or_count


# Binary Operators
class AddOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.add_count = []

    def visit_Add(self, node: cst.Add) -> bool:
        self.add_count.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.add_count

class BitAndOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.bit_and_count = []

    def visit_BitAnd(self, node: cst.BitAnd) -> bool:
        self.bit_and_count.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.bit_and_count

class BitOrOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.bit_or_count = []

    def visit_BitOr(self, node: cst.BitOr) -> bool:
        self.bit_or_count.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.bit_or_count

class BitXorOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.bit_xor_count = []

    def visit_BitXor(self, node: cst.BitXor) -> bool:
        self.bit_xor_count.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.bit_xor_count

class DivideOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.divide_count = []

    def visit_Divide(self, node: cst.Divide) -> bool:
        self.divide_count.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.divide_count

class FloorDivideOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.floor_divide_count = []

    def visit_FloorDivide(self, node: cst.FloorDivide) -> bool:
        self.floor_divide_count.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.floor_divide_count

class LeftShiftOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.left_shift_count = []

    def visit_LeftShift(self, node: cst.LeftShift) -> bool:
        self.left_shift_count.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.left_shift_count

class MatrixMultiplyOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.matrix_multiply_count = []

    def visit_MatrixMultiply(self, node: cst.MatrixMultiply) -> bool:
        self.matrix_multiply_count.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.matrix_multiply_count

class ModuloOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.modulo_count = []

    def visit_Modulo(self, node: cst.Modulo) -> bool:
        self.modulo_count.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.modulo_count

class MultiplyOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.multiply_count = []

    def visit_Multiply(self, node: cst.Multiply) -> bool:
        self.multiply_count.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.multiply_count

class PowerOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.power_count = []

    def visit_Power(self, node: cst.Power) -> bool:
        self.power_count.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.power_count

class RightShiftOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.right_shift_count = []

    def visit_RightShift(self, node: cst.RightShift) -> bool:
        self.right_shift_count.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.right_shift_count

class SubtractOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.subtract_count = []

    def visit_Subtract(self, node: cst.Subtract) -> bool:
        self.subtract_count.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.subtract_count


# Comparison Operators
class EqualOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.equal_count = []

    def visit_Equal(self, node: cst.Equal) -> bool:
        self.equal_count.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.equal_count

class GreaterThanOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.greater_than_count = []

    def visit_GreaterThan(self, node: cst.GreaterThan) -> bool:
        self.greater_than_count.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.greater_than_count

# ... continue with other comparison operators


# Augmented Assignment Operators
class AddAssignOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.add_assign_count = []

    def visit_AddAssign(self, node: cst.AddAssign) -> bool:
        self.add_assign_count.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.add_assign_count

class BitAndAssignOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.bit_and_assign_count = []

    def visit_BitAndAssign(self, node: cst.BitAndAssign) -> bool:
        self.bit_and_assign_count.append(cst.Module([node]).code)
        return True
    def collect(self):
        self.module.visit(self)
        return self.bit_and_assign_count
class BitAndAssignOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.bit_and_assign_count = []

    def visit_BitAndAssign(self, node: cst.BitAndAssign) -> bool:
        self.bit_and_assign_count.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.bit_and_assign_count

class BitOrAssignOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.bit_or_assign_count = []

    def visit_BitOrAssign(self, node: cst.BitOrAssign) -> bool:
        self.bit_or_assign_count.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.bit_or_assign_count

class BitXorAssignOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.bit_xor_assign_count = []

    def visit_BitXorAssign(self, node: cst.BitXorAssign) -> bool:
        self.bit_xor_assign_count.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.bit_xor_assign_count

class DivideAssignOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.divide_assign_count = []

    def visit_DivideAssign(self, node: cst.DivideAssign) -> bool:
        self.divide_assign_count.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.divide_assign_count

class FloorDivideAssignOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.floor_divide_assign_count = []

    def visit_FloorDivideAssign(self, node: cst.FloorDivideAssign) -> bool:
        self.floor_divide_assign_count.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.floor_divide_assign_count

class LeftShiftAssignOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.left_shift_assign_count = []

    def visit_LeftShiftAssign(self, node: cst.LeftShiftAssign) -> bool:
        self.left_shift_assign_count.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.left_shift_assign_count

class MatrixMultiplyAssignOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.matrix_multiply_assign_count = []

    def visit_MatrixMultiplyAssign(self, node: cst.MatrixMultiplyAssign) -> bool:
        self.matrix_multiply_assign_count.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.matrix_multiply_assign_count

class ModuloAssignOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.modulo_assign_count = []

    def visit_ModuloAssign(self, node: cst.ModuloAssign) -> bool:
        self.modulo_assign_count.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.modulo_assign_count

class MultiplyAssignOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.multiply_assign_count = []

    def visit_MultiplyAssign(self, node: cst.MultiplyAssign) -> bool:
        self.multiply_assign_count.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.multiply_assign_count

class PowerAssignOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.power_assign_count = []

    def visit_PowerAssign(self, node: cst.PowerAssign) -> bool:
        self.power_assign_count.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.power_assign_count

class RightShiftAssignOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.right_shift_assign_count = []

    def visit_RightShiftAssign(self, node: cst.RightShiftAssign) -> bool:
        self.right_shift_assign_count.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.right_shift_assign_count

class RightShiftAssignOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.right_shift_assign_count = []

    def visit_RightShiftAssign(self, node: cst.RightShiftAssign) -> bool:
        self.right_shift_assign_count.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.right_shift_assign_count

class SubtractAssignOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.subtract_assign_count = []

    def visit_SubtractAssign(self, node: cst.SubtractAssign) -> bool:
        self.subtract_assign_count.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.subtract_assign_count

# Miscellaneous Operators
class AssignEqualOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.assign_equal_count = []

    def visit_AssignEqual(self, node: cst.AssignEqual) -> bool:
        self.assign_equal_count.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.assign_equal_count

class ColonOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.colon_count = []

    def visit_Colon(self, node: cst.Colon) -> bool:
        self.colon_count.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.colon_count

class CommaOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.comma_count = []

    def visit_Comma(self, node: cst.Comma) -> bool:
        self.comma_count.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.comma_count

class DotOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.dot_count = []

    def visit_Dot(self, node: cst.Dot) -> bool:
        self.dot_count.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.dot_count

class ImportStarOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.import_star_count = []

    def visit_ImportStar(self, node: cst.ImportStar) -> bool:
        self.import_star_count.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.import_star_count

class SemicolonOperatorCollector(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.semicolon_count = []

    def visit_Semicolon(self, node: cst.Semicolon) -> bool:
        self.semicolon_count.append(cst.Module([node]).code)
        return True

    def collect(self):
        self.module.visit(self)
        return self.semicolon_count
