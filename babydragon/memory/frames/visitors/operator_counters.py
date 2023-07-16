import libcst as cst

OPERATOR_COUNTERS = [
    "BitInvertOperatorCounter",
    "MinusOperatorCounter",
    "NotOperatorCounter",
    "PlusOperatorCounter",
    "AndOperatorCounter",
    "OrOperatorCounter",
    "AddOperatorCounter",
    "BitAndOperatorCounter",
    "BitOrOperatorCounter",
    "BitXorOperatorCounter",
    "DivideOperatorCounter",
    "FloorDivideOperatorCounter",
    "LeftShiftOperatorCounter",
    "MatrixMultiplyOperatorCounter",
    "ModuloOperatorCounter",
    "MultiplyOperatorCounter",
    "PowerOperatorCounter",
    "RightShiftOperatorCounter",
    "SubtractOperatorCounter",
    "EqualOperatorCounter",
    "GreaterThanOperatorCounter",
    "GreaterThanEqualOperatorCounter",
    "InOperatorCounter",
    "IsOperatorCounter",
    "LessThanOperatorCounter",
    "LessThanEqualOperatorCounter",
    "NotEqualOperatorCounter",
    "IsNotOperatorCounter",
    "NotInOperatorCounter",
    "AddAssignOperatorCounter",
    "BitAndAssignOperatorCounter",
    "BitOrAssignOperatorCounter",
    "BitXorAssignOperatorCounter",
    "DivideAssignOperatorCounter",
    "FloorDivideAssignOperatorCounter",
    "LeftShiftAssignOperatorCounter",
    "MatrixMultiplyAssignOperatorCounter",
    "ModuloAssignOperatorCounter",
    "MultiplyAssignOperatorCounter",
    "PowerAssignOperatorCounter",
    "RightShiftAssignOperatorCounter",
    "SubtractAssignOperatorCounter",
    "AssignEqualOperatorCounter",
    "ColonOperatorCounter",
    "CommaOperatorCounter",
    "DotOperatorCounter",
]


# Unary Operators
class BitInvertOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.bit_invert_operator_count = 0

    def visit_BitInvert(self, node: cst.BitInvert) -> bool:
        self.bit_invert_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.bit_invert_operator_count

class MinusOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.minus_operator_count = 0

    def visit_Minus(self, node: cst.Minus) -> bool:
        self.minus_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.minus_operator_count

class NotOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.not_operator_count = 0

    def visit_Not(self, node: cst.Not) -> bool:
        self.not_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.not_operator_count

class PlusOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.plus_operator_count = 0

    def visit_Plus(self, node: cst.Plus) -> bool:
        self.plus_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.plus_operator_count

# Boolean Operators
class AndOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.and_operator_count = 0

    def visit_And(self, node: cst.And) -> bool:
        self.and_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.and_operator_count

class OrOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.or_operator_count = 0

    def visit_Or(self, node: cst.Or) -> bool:
        self.or_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.or_operator_count

# Binary Operators
class AddOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.add_operator_count = 0

    def visit_Add(self, node: cst.Add) -> bool:
        self.add_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.add_operator_count

# Binary Operators
class BitAndOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.bit_and_operator_count = 0

    def visit_BitAnd(self, node: cst.BitAnd) -> bool:
        self.bit_and_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.bit_and_operator_count

class BitOrOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.bit_or_operator_count = 0

    def visit_BitOr(self, node: cst.BitOr) -> bool:
        self.bit_or_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.bit_or_operator_count

class BitXorOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.bit_xor_operator_count = 0

    def visit_BitXor(self, node: cst.BitXor) -> bool:
        self.bit_xor_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.bit_xor_operator_count

class DivideOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.divide_operator_count = 0

    def visit_Divide(self, node: cst.Divide) -> bool:
        self.divide_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.divide_operator_count

class FloorDivideOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.floor_divide_operator_count = 0

    def visit_FloorDivide(self, node: cst.FloorDivide) -> bool:
        self.floor_divide_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.floor_divide_operator_count

class LeftShiftOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.left_shift_operator_count = 0

    def visit_LeftShift(self, node: cst.LeftShift) -> bool:
        self.left_shift_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.left_shift_operator_count

class MatrixMultiplyOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.matrix_multiply_operator_count = 0

    def visit_MatrixMultiply(self, node: cst.MatrixMultiply) -> bool:
        self.matrix_multiply_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.matrix_multiply_operator_count

class ModuloOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.modulo_operator_count = 0

    def visit_Modulo(self, node: cst.Modulo) -> bool:
        self.modulo_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.modulo_operator_count

class MultiplyOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.multiply_operator_count = 0

    def visit_Multiply(self, node: cst.Multiply) -> bool:
        self.multiply_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.multiply_operator_count

class PowerOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.power_operator_count = 0

    def visit_Power(self, node: cst.Power) -> bool:
        self.power_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.power_operator_count

class RightShiftOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.right_shift_operator_count = 0

    def visit_RightShift(self, node: cst.RightShift) -> bool:
        self.right_shift_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.right_shift_operator_count

class SubtractOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.subtract_operator_count = 0

    def visit_Subtract(self, node: cst.Subtract) -> bool:
        self.subtract_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.subtract_operator_count


# Comparison Operators
class EqualOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.equal_operator_count = 0

    def visit_Equal(self, node: cst.Equal) -> bool:
        self.equal_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.equal_operator_count

# Comparison Operators
class EqualOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.equal_operator_count = 0

    def visit_Equal(self, node: cst.Equal) -> bool:
        self.equal_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.equal_operator_count

class GreaterThanOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.greater_than_operator_count = 0

    def visit_GreaterThan(self, node: cst.GreaterThan) -> bool:
        self.greater_than_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.greater_than_operator_count

class GreaterThanEqualOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.greater_than_equal_operator_count = 0

    def visit_GreaterThanEqual(self, node: cst.GreaterThanEqual) -> bool:
        self.greater_than_equal_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.greater_than_equal_operator_count

class InOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.in_operator_count = 0

    def visit_In(self, node: cst.In) -> bool:
        self.in_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.in_operator_count

class IsOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.is_operator_count = 0

    def visit_Is(self, node: cst.Is) -> bool:
        self.is_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.is_operator_count

class LessThanOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.less_than_operator_count = 0

    def visit_LessThan(self, node: cst.LessThan) -> bool:
        self.less_than_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.less_than_operator_count

class LessThanEqualOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.less_than_equal_operator_count = 0

    def visit_LessThanEqual(self, node: cst.LessThanEqual) -> bool:
        self.less_than_equal_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.less_than_equal_operator_count

class NotEqualOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.not_equal_operator_count = 0

    def visit_NotEqual(self, node: cst.NotEqual) -> bool:
        self.not_equal_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.not_equal_operator_count

class IsNotOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.is_not_operator_count = 0

    def visit_IsNot(self, node: cst.IsNot) -> bool:
        self.is_not_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.is_not_operator_count

class NotInOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.not_in_operator_count = 0

    def visit_NotIn(self, node: cst.NotIn) -> bool:
        self.not_in_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.not_in_operator_count


# Augmented Assignment Operators
class AddAssignOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.add_assign_operator_count = 0

    def visit_AddAssign(self, node: cst.AddAssign) -> bool:
        self.add_assign_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.add_assign_operator_count

class BitAndAssignOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.bit_and_assign_operator_count = 0

    def visit_BitAndAssign(self, node: cst.BitAndAssign) -> bool:
        self.bit_and_assign_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.bit_and_assign_operator_count

class BitOrAssignOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.bit_or_assign_operator_count = 0

    def visit_BitOrAssign(self, node: cst.BitOrAssign) -> bool:
        self.bit_or_assign_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.bit_or_assign_operator_count

class BitXorAssignOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.bit_xor_assign_operator_count = 0

    def visit_BitXorAssign(self, node: cst.BitXorAssign) -> bool:
        self.bit_xor_assign_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.bit_xor_assign_operator_count

class DivideAssignOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.divide_assign_operator_count = 0

    def visit_DivideAssign(self, node: cst.DivideAssign) -> bool:
        self.divide_assign_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.divide_assign_operator_count

class FloorDivideAssignOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.floor_divide_assign_operator_count = 0

    def visit_FloorDivideAssign(self, node: cst.FloorDivideAssign) -> bool:
        self.floor_divide_assign_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.floor_divide_assign_operator_count

class LeftShiftAssignOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.left_shift_assign_operator_count = 0

    def visit_LeftShiftAssign(self, node: cst.LeftShiftAssign) -> bool:
        self.left_shift_assign_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.left_shift_assign_operator_count

class MatrixMultiplyAssignOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.matrix_multiply_assign_operator_count = 0

    def visit_MatrixMultiplyAssign(self, node: cst.MatrixMultiplyAssign) -> bool:
        self.matrix_multiply_assign_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.matrix_multiply_assign_operator_count

class ModuloAssignOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.modulo_assign_operator_count = 0

    def visit_ModuloAssign(self, node: cst.ModuloAssign) -> bool:
        self.modulo_assign_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.modulo_assign_operator_count

class MultiplyAssignOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.multiply_assign_operator_count = 0

    def visit_MultiplyAssign(self, node: cst.MultiplyAssign) -> bool:
        self.multiply_assign_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.multiply_assign_operator_count

class PowerAssignOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.power_assign_operator_count = 0

    def visit_PowerAssign(self, node: cst.PowerAssign) -> bool:
        self.power_assign_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.power_assign_operator_count

class RightShiftAssignOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.right_shift_assign_operator_count = 0

    def visit_RightShiftAssign(self, node: cst.RightShiftAssign) -> bool:
        self.right_shift_assign_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.right_shift_assign_operator_count

class SubtractAssignOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.subtract_assign_operator_count = 0

    def visit_SubtractAssign(self, node: cst.SubtractAssign) -> bool:
        self.subtract_assign_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.subtract_assign_operator_count


# Miscellaneous Operators
class AssignEqualOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.assign_equal_operator_count = 0

    def visit_AssignEqual(self, node: cst.AssignEqual) -> bool:
        self.assign_equal_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.assign_equal_operator_count

# Miscellaneous Operators
class AssignEqualOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.assign_equal_operator_count = 0

    def visit_AssignEqual(self, node: cst.AssignEqual) -> bool:
        self.assign_equal_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.assign_equal_operator_count

class ColonOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.colon_operator_count = 0

    def visit_Colon(self, node: cst.Colon) -> bool:
        self.colon_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.colon_operator_count

class CommaOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.comma_operator_count = 0

    def visit_Comma(self, node: cst.Comma) -> bool:
        self.comma_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.comma_operator_count

class DotOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.dot_operator_count = 0

    def visit_Dot(self, node: cst.Dot) -> bool:
        self.dot_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.dot_operator_count

class ImportStarOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.import_star_operator_count = 0

    def visit_ImportStar(self, node: cst.ImportStar) -> bool:
        self.import_star_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.import_star_operator_count

class SemicolonOperatorCounter(cst.CSTVisitor):
    def __init__(self, code: str):
        self.module = cst.parse_module(code)
        self.semicolon_operator_count = 0

    def visit_Semicolon(self, node: cst.Semicolon) -> bool:
        self.semicolon_operator_count += 1
        return True

    def collect(self):
        self.module.visit(self)
        return self.semicolon_operator_count
