from typing import Tuple


class OutOfDomain(Exception):
    def __init__(self, value: int | float, domain: Tuple[int | float | str, int | float | str] = (0, "inf"),
                 brackets: str = "()") -> None:
        message = f"Argument {value} contains a value outside of the function domain {brackets[0]}{domain[0]}, " \
                  f"{domain[1]}{brackets[1]}"
        super().__init__(message)
