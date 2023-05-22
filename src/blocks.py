from abc import abstractmethod
from math import comb
from typing import Callable

import torch
import torch.nn as nn

from errors import OutOfDomain

ZERO_CORRECTION = 1e-9


class BlockInterface(nn.Module):
    def __init__(self, in_features: int, out_features: int, *args, **kwargs) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def scalar_weight(self) -> float:
        pass


class WeightedBlockInterface(BlockInterface):
    def __init__(self, in_features: int, out_features: int, *args, **kwargs) -> None:
        super().__init__(in_features=in_features, out_features=out_features)
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
        if self.in_features == 1 and self.out_features == 1:
            return f"{float(self.weight):2.4f}"
        return "[" + ", ".join(["[" + ", ".join([eval("f'{wi:2.4f}'") for wi in w]) + "]" for w in self.weight]) + "]"

    def str(self, inner: str = "x") -> str:
        return f"{type(self).__name__[:-5].lower()}({inner}) * {self.weight_str}"

    def __str__(self) -> str:
        return self.str("x")


class LinearBlock(WeightedBlockInterface):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clone().mm(self.weight)

    def str(self, inner: str = "x") -> str:
        return f"{inner} * {self.weight_str}"


class InverseBlock(WeightedBlockInterface):
    def __init__(self, in_features: int, out_features: int, correct_values: bool = True, *args, **kwargs) -> None:
        super().__init__(in_features=in_features, out_features=out_features, *args, **kwargs)
        self.correct_values = correct_values

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.sum(x == 0):
            if self.correct_values:
                x[x == 0] = ZERO_CORRECTION
            else:
                raise OutOfDomain(x)
        return (1 / x).mm(self.weight)

    def str(self, inner: str = "x") -> str:
        return f"(1 / {inner}) * {self.weight_str}"


class BiasBlock(BlockInterface):
    def __init__(self, in_features: int, out_features: int, *args, **kwargs) -> None:
        super().__init__(in_features=in_features, out_features=out_features, *args, **kwargs)
        self.bias = nn.Parameter(torch.rand(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bias

    def str(self, *args, **kwargs) -> str:
        if self.in_features == 1 and self.out_features == 1:
            return ", ".join([eval("f'{b:2.4f}'") for b in self.bias])
        return "[" + ", ".join([eval("f'{b:2.4f}'") for b in self.bias]) + "]"

    def __str__(self) -> str:
        return self.str()

    @property
    def scalar_weight(self) -> torch.Tensor:
        return torch.mean(self.bias).detach().cpu()


class SinBlock(WeightedBlockInterface):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x).mm(self._weight)


class CosBlock(WeightedBlockInterface):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cos(x).mm(self._weight)


class AbsBlock(WeightedBlockInterface):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x).mm(self._weight)


class SigmoidBlock(WeightedBlockInterface):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.sigmoid(x).mm(self._weight)


class LogBlockInterface(WeightedBlockInterface):
    def __init__(self, in_features: int, out_features: int, log_mode: str = "symmetric", *args, **kwargs) -> None:
        super().__init__(in_features=in_features, out_features=out_features, *args, **kwargs)
        self._set_inner(log_mode)

    def _set_inner(self, log_mode: str) -> None:
        match log_mode:
            case "symmetric":
                self._inner_forward = self._forward_symmetric
            case "domain_adaptation":
                self._domain_minimum = None
                self._inner_forward = self._forward_domain_adaptation
            case "constant":
                self._inner_forward = self._forward_constant

    @property
    @abstractmethod
    def log(self) -> Callable[[torch.Tensor], torch.Tensor]:
        pass

    def _forward_domain_adaptation(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        if self.training:
            self._domain_minimum = x.min(-2).values if self._domain_minimum is None else self._domain_minimum
            self._domain_minimum = torch.min(self._domain_minimum, x.min(-2).values)
        elif torch.any(x.min(-2).value <= self._domain_minimum):
            raise OutOfDomain(x, domain=(self._domain_minimum, "inf"))
        x -= self._domain_minimum - 1e-1
        return self.log(x).mm(self._weight)

    def _forward_constant(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        x[x <= 0] = 1
        return self.log(x).mm(self._weight)

    def _forward_symmetric(self, x: torch.Tensor) -> torch.Tensor:
        zero_mask = (x == 0)
        negative_mask = (x < 0)
        x[zero_mask] = 1
        x = self.log(torch.abs(x)).mm(self._weight)
        x[zero_mask] = 0
        x[negative_mask] *= -1
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._inner_forward(x)


class LnBlock(LogBlockInterface):
    @property
    def log(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return torch.log


class Log2Block(LogBlockInterface):
    @property
    def log(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return torch.log2


class Log10Block(LogBlockInterface):
    @property
    def log(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return torch.log10


class PowInterface(WeightedBlockInterface):
    @property
    @abstractmethod
    def exponent(self) -> int:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.pow(self.exponent).mm(self._weight)

    def str(self, inner: str = "x") -> str:
        return f"({inner}) ^ {self.exponent} * {self.weight_str}"


class Pow2Block(PowInterface):
    @property
    def exponent(self) -> int:
        return 2


class Pow3Block(PowInterface):
    @property
    def exponent(self) -> int:
        return 3


class Pow4Block(PowInterface):
    @property
    def exponent(self) -> int:
        return 4


class Pow5Block(PowInterface):
    @property
    def exponent(self) -> int:
        return 5


class PolynomialBlockInterface(WeightedBlockInterface):
    def __init__(self, in_features: int, out_features: int, *args, **kwargs) -> None:
        super().__init__(in_features=in_features, out_features=out_features)
        self._comb_num = comb(in_features, self.n_pairs)
        self._weight = nn.Parameter(torch.rand(self._comb_num, out_features))

    @property
    @abstractmethod
    def n_pairs(self) -> int:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idcs = torch.combinations(torch.arange(0, x.shape[1]), r=self.n_pairs)
        return x[:, idcs].prod(-1).mm(self._weight)


class Polynomial2Block(PolynomialBlockInterface):
    @property
    def n_pairs(self) -> int:
        return 2

    def str(self, inner: str = "x") -> str:
        return f"poly({inner}, 2) * {self.weight_str}"


class Polynomial3Block(PolynomialBlockInterface):
    @property
    def n_pairs(self) -> int:
        return 3

    def str(self, inner: str = "x") -> str:
        return f"poly({inner}, 2) * {self.weight_str}"


ALL_BLOCKS = [AbsBlock, BiasBlock, CosBlock, InverseBlock, LnBlock, Log2Block, Log10Block, LinearBlock,
              SigmoidBlock, SinBlock, Pow2Block, Pow3Block, Pow4Block, Pow5Block, Polynomial2Block, Polynomial3Block]
