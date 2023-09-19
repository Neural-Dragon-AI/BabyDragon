from typing import List, Optional, Tuple, Union

from pydantic import Field, field_validator

from babydragon.types.base import BDType


class RealData(BDType):
    range: Optional[Tuple[Union[float, int], Union[float, int]]] = Field(
        None, description="An optional inclusive range (min, max) for the value."
    )
    value: Union[float, int] = Field(
        ..., description="The real value data. It should be a float or an integer."
    )

    @field_validator("value")
    def validate_value(cls, v, values):
        value_range = values.data["range"]
        if value_range is not None:
            min_value, max_value = value_range
            if not min_value <= v <= max_value:
                raise ValueError(
                    f"Value {v} is not within the specified range {value_range}."
                )
        return v


class RealDataList(BDType):
    range: Optional[Tuple[Union[float, int], Union[float, int]]] = Field(
        None, description="An optional inclusive range (min, max) for the values."
    )
    values: List[RealData] = Field(
        ...,
        description="The list of real value data. Each should be a RealeData object.",
    )

    @field_validator("values")
    def validate_values(cls, values, values_dict):
        list_range = values_dict.data.get("range")
        if list_range is not None:
            min_value, max_value = list_range
            for value in values:
                if not min_value <= value.value <= max_value:
                    raise ValueError(
                        f"Value {value.value} of RealData object is not within the specified range {list_range}."
                    )
        return values


class MultiDimensionalReal(BDType):
    range: Optional[
        Union[
            Tuple[Union[float, int], Union[float, int]],
            List[Tuple[Union[float, int], Union[float, int]]],
        ]
    ] = Field(
        None,
        description="An optional inclusive range (min, max) for the values. If a tuple, applies to all dimensions. If a list, it must match the dimension length.",
    )
    values: List[RealData] = Field(
        ...,
        description="The list of real data for each dimension. Each should be a RealData object.",
    )

    @field_validator("values")
    def validate_values(cls, values, values_dict):
        range_values = values_dict.data.get("range")
        if range_values is not None:
            # If range is a tuple, apply it to all dimensions
            if isinstance(range_values, tuple):
                min_value, max_value = range_values
                for value in values:
                    if not min_value <= value.value <= max_value:
                        raise ValueError(
                            f"Value {value.value} of RealData object is not within the specified range {range_values}."
                        )
            # If range is a list, it must have the same length as values
            elif isinstance(range_values, list):
                if len(values) != len(range_values):
                    raise ValueError(
                        "If range is a list, it must have the same length as values."
                    )
                for value, (min_value, max_value) in zip(values, range_values):
                    if not min_value <= value.value <= max_value:
                        raise ValueError(
                            f"Value {value.value} of RealData object is not within the specified range ({min_value}, {max_value})."
                        )
        return values


class MultiDimensionalRealList(BDType):
    range: Optional[
        Union[
            Tuple[Union[float, int], Union[float, int]],
            List[Tuple[Union[float, int], Union[float, int]]],
        ]
    ] = Field(
        None,
        description="An optional inclusive range (min, max) for the values in all dimensions. If a tuple, applies to all dimensions. If a list, it must match the dimension length.",
    )
    values: List[MultiDimensionalReal] = Field(
        ...,
        description="The list of multi-dimensional real data. Each should be a MultiDimensionalReal object.",
    )

    @field_validator("values")
    def validate_values(cls, values, values_dict):
        range_values = values_dict.data.get("range")
        dimension_length = len(values[0].values) if values else 0
        if range_values is not None:
            # If range is a tuple, apply it to all dimensions
            if isinstance(range_values, tuple):
                min_value, max_value = range_values
                for multi_real in values:
                    if len(multi_real.values) != dimension_length:
                        raise ValueError(
                            "All MultiDimensionalReal in the list must have the same length."
                        )
                    for value in multi_real.values:
                        if not min_value <= value.value <= max_value:
                            raise ValueError(
                                f"Value {value.value} of RealData object is not within the specified range {range_values}."
                            )
            # If range is a list, it must have the same length as values in each dimension
            elif isinstance(range_values, list):
                if len(range_values) != dimension_length:
                    raise ValueError(
                        "If range is a list, it must have the same length as values in each dimension."
                    )
                for multi_real in values:
                    if len(multi_real.values) != dimension_length:
                        raise ValueError(
                            "All MultiDimensionalReal in the list must have the same length."
                        )
                    for value, (min_value, max_value) in zip(
                        multi_real.values, range_values
                    ):
                        if not min_value <= value.value <= max_value:
                            raise ValueError(
                                f"Value {value.value} of RealData object is not within the specified range ({min_value}, {max_value})."
                            )
        return values
