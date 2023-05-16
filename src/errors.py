from typing import Tuple


class OutOfDomain(Exception):
    def __init__(self, value: int | float, domain: Tuple[int | float | str, int | float | str] = (0, "inf"),
                 brackets: str = "()") -> None:
        message = f"Argument {value} contains a value outside of the function domain {brackets[0]}{domain[0]}, " \
                  f"{domain[1]}{brackets[1]}"
        super().__init__(message)


class TooFewInputs(Exception):
    def __init__(self, input_size: int, min_size: int, block_name: str) -> None:
        message = f"The input composed of {input_size} variables is too small for {block_name} that requires {min_size} " \
                  "variables"
        super().__init__(message)
