from abc import ABC
from typing import Dict, List, Optional, Tuple, Type, Union

from blocks import *


class Model(nn.Module):
    def __init__(self, blocks: List[BlockInterface]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_hat = self.blocks[0](x)
        for b in self.blocks[1:]:
            y_hat += b(x)
        return y_hat

    @property
    def weights(self) -> List[torch.Tensor | None]:
        return [b.weight if isinstance(b, WeightedBlockInterface) else None for b in self.blocks]

    @property
    def scalar_weights(self) -> List[torch.Tensor]:
        return [b.scalar_weight for b in self.blocks]

    @property
    def least_significant(self) -> Tuple[int, float]:
        idx = 0
        least_significance = float("inf")
        for i_sw, sw in enumerate(self.scalar_weights):
            if torch.abs(sw) < least_significance:
                idx = i_sw
                least_significance = torch.abs(sw)
        return idx, least_significance

    def delitem(self, idx: int) -> None:
        del self.blocks[idx]

    def str(self, inner: str = "x") -> str:
        return " + ".join([b.str(inner) for b in self.blocks])

    def __str__(self) -> str:
        return "y = " + self.str("x")

    def __len__(self) -> int:
        return len(self.blocks)


class MultiLayerModel(Model):
    def __init__(self, block: WeightedBlockInterface, blocks: Union[List[WeightedBlockInterface], List[Model]]) -> None:
        super().__init__(blocks)
        self.block = block

    @property
    def scalar_weight(self) -> torch.Tensor:
        return self.block.scalar_weight

    def str(self, inner: str = "x") -> str:
        return self.block.str(inner="(" + " + ".join([b.str() for b in self.blocks]) + ")")

    def __str__(self) -> str:
        return "y = " + self.str()

    def __len__(self) -> int:
        length = 0
        for b in self.blocks:
            try:
                length += len(b)
            except:
                length += 1
        return length + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_hat = self.blocks[0](x)
        for b in self.blocks[1:]:
            y_hat += b(x)
        return self.block(y_hat)


class ModelFactoryInterface(ABC):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, max_size: Optional[int] = None, epochs: int = 1000,
                 early_stopping: int = 5, lr: float = 1e-2, criterion_cls: nn.Module = nn.MSELoss, **kwargs) -> None:
        self.in_features = x.shape[1]
        self.out_features = y.shape[1]
        self.domain_min = x.min(-2).values
        self.x = x
        self.y = y
        self.max_size = max_size
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.lr = lr
        self.criterion = criterion_cls()
        self.kwargs = kwargs

    def fit(self, model: Model) -> Model:
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_history = []
        for _ in range(self.epochs):
            optimizer.zero_grad()
            y_hat = model(self.x)
            loss = self.criterion(y_hat, self.y)
            loss.backward()
            optimizer.step()
            condition, loss_history = self.check_early_stopping(loss.detach().item(), loss_history)
            if condition:
                break
        return model

    def check_early_stopping(self, loss: float, loss_hist: List[float]) -> Tuple[bool, List[float]]:
        loss_hist.append(loss)
        if len(loss_hist) > self.early_stopping + 1:
            _ = loss_hist.pop(0)
            if loss >= max(loss_hist[:-1]):
                return True, loss_hist
        return False, loss_hist

    def from_occurrence_dict(self, blocks_classes: Dict[Type[BlockInterface], int]) -> Model:
        blocks = []
        for (cls, occurrence) in blocks_classes.items():
            blocks.extend(
                cls(self.in_features, self.out_features, **self.kwargs) for _ in range(occurrence))
        model = Model(blocks=blocks)
        model = self.fit_prune(model)
        return model


class ModelFactory(ModelFactoryInterface):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, max_size: Optional[int] = None, epochs: int = 1000,
                 early_stopping: int = 5, lr: float = 1e-2, criterion_cls: nn.Module = nn.MSELoss, **kwargs) -> None:
        super().__init__(x, y, max_size, epochs, early_stopping, lr, criterion_cls, **kwargs)

    def prune(self, model: Model) -> Model:
        idx, _ = model.least_significant
        model.delitem(idx)
        return model

    def fit_prune(self, model: Model) -> Model:
        if self.max_size is None:
            return model
        while len(model.blocks) > self.max_size:
            model = self.fit(model)
            model = self.prune(model)
        model = self.fit(model)
        return model

    def from_class_list(self, blocks_classes: List[Type[BlockInterface]]) -> Model:
        model = Model(
            blocks=[bc(self.in_features, self.out_features, **self.kwargs) for bc in blocks_classes])
        model = self.fit_prune(model)
        return model


class MultiLayerModelFactory(ModelFactoryInterface):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, max_size: Optional[int] = None, layers: int = 1,
                 epochs: int = 1000, early_stopping: int = 5, lr: float = 1e-2,
                 criterion_cls: nn.Module = nn.MSELoss, **kwargs) -> None:
        super().__init__(x, y, max_size, epochs, early_stopping, lr, criterion_cls, **kwargs)
        self.layers = layers

    def prune(self, model: Model) -> Union[Model, Tuple[Model, bool]]:
        if isinstance(model.blocks[0], BlockInterface):
            if len(model.blocks) > self.max_size:
                idx, _ = model.least_significant
                model.delitem(idx)
                return model, True
            return model, False
        else:
            news = []
            for b_i, block in enumerate(model.blocks):
                new, status = self.prune(block)
                news.append(new)
            if status:
                model.blocks = nn.ModuleList(news)
            else:
                idx, _ = model.least_significant
                model.delitem(idx)
            return model, True

    def fit_prune(self, model: Model) -> Model:
        if self.max_size is None:
            return model
        while len(model.blocks) > self.max_size:
            model = self.fit(model)
            model, _ = self.prune(model)
        model = self.fit(model)
        return model

    def from_class_list(self, blocks_classes: List[Type[WeightedBlockInterface]],
                        block: Type[WeightedBlockInterface] = LinearBlock,
                        current_layer: int = 1) -> Model:
        if self.layers == current_layer:
            model = MultiLayerModel(
                block=block(self.out_features, self.out_features, **self.kwargs),
                blocks=[bc(self.in_features, self.out_features, **self.kwargs) for bc in blocks_classes])
        else:
            model = MultiLayerModel(
                block=block(self.out_features, self.out_features, **self.kwargs),
                blocks=[self.from_class_list(blocks_classes, block=bc, current_layer=current_layer + 1) for bc in
                        blocks_classes]
            )
        if current_layer == 1:
            model = self.fit_prune(model)
        return model
