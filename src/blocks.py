from abc import abstractmethod
from typing import Callable

import torch
import torch.nn as nn

from errors import OutOfDomain

ZERO_CORRECTION = 1e-9


class AbstractBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def scalar_weight(self) -> float:
        pass


class AbstractWeightedBlock(AbstractBlock):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.rand(in_features, out_features))
        self.sigmoid = nn.Sigmoid()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @property
    def scalar_weight(self) -> torch.Tensor:
        return torch.mean(self.weight)


class LinearBlock(AbstractWeightedBlock):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features=in_features, out_features=out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x


class InverseBlock(AbstractWeightedBlock):
    def __init__(self, in_features: int, out_features: int, correct_values: bool = True) -> None:
        super().__init__(in_features=in_features, out_features=out_features)
        self.correct_values = correct_values

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.sum(x == 0):
            if self.correct_values:
                x[x == 0] = ZERO_CORRECTION
            else:
                raise OutOfDomain(x)
        return self.weight * (1 / x)


class BiasBlock(AbstractBlock):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.rand(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bias

    @property
    def scalar_weight(self) -> torch.Tensor:
        return torch.mean(self.bias)


class SinBlock(AbstractWeightedBlock):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features=in_features, out_features=out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * torch.sin(x)


class CosBlock(AbstractWeightedBlock):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features=in_features, out_features=out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * torch.cos(x)


class AbsBlock(AbstractWeightedBlock):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features=in_features, out_features=out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * torch.abs(x)


class SigmoidBlock(AbstractWeightedBlock):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features=in_features, out_features=out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * nn.functional.sigmoid(x)


class AbstractLogBlock(AbstractWeightedBlock):
    def __init__(self, in_features: int, out_features: int, domain_minimum: int | float = 0.0) -> None:
        super().__init__(in_features=in_features, out_features=out_features)
        self.domain_minimum = domain_minimum

    @property
    @abstractmethod
    def log(self) -> Callable[[torch.Tensor], torch.Tensor]:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x += self.domain_minimum
        if torch.sum(x <= 0):
            raise OutOfDomain(x)
        return self.weight * self.log(x)


class LnBlock(AbstractLogBlock):
    def __init__(self, in_features: int, out_features: int, domain_minimum: int | float = 0.0) -> None:
        super().__init__(in_features=in_features, out_features=out_features, domain_minimum=domain_minimum)

    @property
    def log(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return torch.log


class Log2Block(AbstractLogBlock):
    def __init__(self, in_features: int, out_features: int, domain_minimum: int | float = 0.0) -> None:
        super().__init__(in_features=in_features, out_features=out_features, domain_minimum=domain_minimum)

    @property
    def log(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return torch.log2


class Log10Block(AbstractLogBlock):
    def __init__(self, in_features: int, out_features: int, domain_minimum: int | float = 0.0) -> None:
        super().__init__(in_features=in_features, out_features=out_features, domain_minimum=domain_minimum)

    @property
    def log(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return torch.log10
