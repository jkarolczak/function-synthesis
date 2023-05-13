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
        self._weight = nn.Parameter(torch.rand(in_features, out_features))

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @property
    def weight(self) -> torch.Tensor:
        return self._weight.data

    @property
    def scalar_weight(self) -> torch.Tensor:
        return torch.mean(self._weight).detach().cpu()

    @property
    def weight_str(self) -> str:
        return "[" + ", ".join(["[" + ", ".join([eval("f'{wi:2.4f}'") for wi in w]) + "]" for w in self.weight]) + "]"

    def str(self, inner: str = "x") -> str:
        return f"{self.weight_str} * {type(self).__name__[:-5].lower()}({inner})"

    def __str__(self) -> str:
        return self.str("x")


class LinearBlock(AbstractWeightedBlock):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features=in_features, out_features=out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._weight * x.clone()

    def str(self, inner: str = "x") -> str:
        return f"{self.weight_str} * {inner}"


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
        return self._weight * (1 / x)

    def str(self, inner: str = "x") -> str:
        return f"{self.weight_str} * (1 / {inner})"


class BiasBlock(AbstractBlock):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.rand(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bias

    def str(self, *args, **kwargs) -> str:
        return "[" + ", ".join([eval("f'{b:2.4f}'") for b in self.bias]) + "]"

    def __str__(self) -> str:
        return self.str()

    @property
    def scalar_weight(self) -> torch.Tensor:
        return torch.mean(self.bias).detach().cpu()


class SinBlock(AbstractWeightedBlock):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features=in_features, out_features=out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._weight * torch.sin(x)


class CosBlock(AbstractWeightedBlock):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features=in_features, out_features=out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._weight * torch.cos(x)


class AbsBlock(AbstractWeightedBlock):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features=in_features, out_features=out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._weight * torch.abs(x)


class SigmoidBlock(AbstractWeightedBlock):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features=in_features, out_features=out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._weight * nn.functional.sigmoid(x)


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
        return self._weight * self.log(x)


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


ALL_BLOCKS = [AbsBlock, BiasBlock, CosBlock, InverseBlock, LnBlock, Log2Block, Log10Block, LinearBlock, SigmoidBlock,
              SinBlock]
