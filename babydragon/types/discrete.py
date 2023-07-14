from pydantic import BaseModel, Field, FieldValidationInfo, field_validator
from typing import Union, List, Optional, Dict
from babydragon.types.base import BDType


class DiscreteDataInt(BDType):
    alphabet: Optional[Set[int]] = Field(None, description="Set of allowed discrete variables. All elements should be integers.")
    value: int = Field(..., description="The discrete data value. It should be an integer.")

    @field_validator('alphabet')
    def check_alphabet(cls, v):
        if not all(isinstance(item, int) for item in v):
            raise ValueError("All elements in 'alphabet' should be integers.")
        return v

    @field_validator('value')
    def check_value(cls, v, info: FieldValidationInfo):
        alphabet = info.data.get('alphabet')
        if alphabet is not None and v not in alphabet:
            raise ValueError("Value must be in the alphabet.")
        return v


class DiscreteDataStr(BDType):
    alphabet: Optional[Set[str]] = Field(None, description="Set of allowed discrete variables. All elements should be strings.")
    value: str = Field(..., description="The discrete data value. It should be a string.")

    @field_validator('alphabet')
    def check_alphabet(cls, v):
        if not all(isinstance(item, str) for item in v):
            raise ValueError("All elements in 'alphabet' should be strings.")
        return v

    @field_validator('value')
    def check_value(cls, v, info: FieldValidationInfo):
        alphabet = info.data.get('alphabet')
        if alphabet is not None and v not in alphabet:
            raise ValueError("Value must be in the alphabet.")
        return v


# The DiscreteDataInt and DiscreteDataStr models are defined as before

class DiscreteDataList(BaseModel):
    alphabet: Optional[Set[Union[int, str]]] = Field(None, description="Set of allowed discrete variables. All elements should be of the same type (either integers or strings).")
    value: List[Union[DiscreteDataInt, DiscreteDataStr]] = Field(..., description="The list of discrete data values. All elements should be either DiscreteDataInt or DiscreteDataStr, not a mix.")

    @field_validator('value')
    def check_alphabets(cls, value, info: FieldValidationInfo):
        list_alphabet = info.data.get('alphabet')
        if list_alphabet is not None:
            for item in value:
                item_alphabet = item.alphabet
                if item_alphabet is not None and not set(item_alphabet).issubset(list_alphabet):
                    raise ValueError(f"Item alphabet {item_alphabet} is not a subset of the list alphabet {list_alphabet}.")
        return value
class MultiDimensionalDiscrete(BDType):
    value: List[Union[DiscreteDataInt, DiscreteDataStr]] = Field(..., description="The multidimensional discrete data value. It should be a list of either DiscreteDataInt or DiscreteDataStr.")
    type_dictionary: Dict[int, str] = Field(default_factory=dict, description="The helper dictionary containing the type of each dimension of the value list.")
    def __init__(self, **data):
        super().__init__(**data)
        self.type_dictionary = {i: item.__class__.__name__ for i, item in enumerate(self.value)}
        
    @field_validator('value')
    def check_value(cls, value):
        if len(value) < 2:
            raise ValueError("For multidimensional discrete data, size of the list should be at least 2. For less than 2, use DiscreteDataInt or DiscreteDataStr.")
        return value
    
    from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Union, Tuple, Any, Set

class MultiDimensionalDiscreteList(BDType):
    values: List[MultiDimensionalDiscrete] = Field(..., description="The list of multidimensional discrete data values. All elements should be instances of MultiDimensionalDiscrete.")
    joint_alphabet: Optional[Set[Tuple[Any, ...]]] = Field(None, description="Set of tuples representing allowed discrete variable combinations. All elements should be tuples of the same length as the number of dimensions in each joint discrete variable.")

    @field_validator('values')
    def check_type_dictionaries(cls, values):
        first_type_dictionary = values[0].type_dictionary
        for value in values[1:]:
            if value.type_dictionary != first_type_dictionary:
                raise ValueError("All elements in 'values' should have the same 'type_dictionary'.")
        return values

    @field_validator('joint_alphabet')
    def check_joint_alphabet(cls, v, info):
        if v is not None and "values" in info.data:
            expected_tuple_length = len(info.data["values"][0].value)
            for item in v:
                if not isinstance(item, tuple) or len(item) != expected_tuple_length:
                    raise ValueError(f"Each element in 'joint_alphabet' should be a tuple of length {expected_tuple_length}.")
                for dim_value, dim_alphabet in zip(item, [value.alphabet for value in info.data["values"][0].value]):
                    if dim_alphabet is not None and dim_value not in dim_alphabet:
                        raise ValueError(f"Value {dim_value} is not in the alphabet for its dimension.")
        return v


    