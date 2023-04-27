from typing import Dict, List, Optional, Tuple, Type

from blocks import *


class Model(nn.Module):
    def __init__(self, blocks: List[AbstractBlock]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_hat = torch.zeros((x.shape[0], 1))
        for b in self.blocks:
            y_hat += b(x)
        return y_hat

    @property
    def weights(self) -> List[torch.Tensor | None]:
        return [b.weight if isinstance(b, AbstractWeightedBlock) else None for b in self.blocks]

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

    def delitem(self, idx: int) -> torch.Tensor:
        del self.blocks[idx]


class MultiLayerModel(nn.Module):
    def __init__(self, block: AbstractWeightedBlock, blocks: List[AbstractWeightedBlock]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.block = block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_hat = torch.zeros((x.shape[0], 1))
        for b in self.blocks:
            y_hat += b(x)
        return self.block(y_hat)

    @property
    def weights(self) -> List[torch.Tensor | None]:
        return [b.weight if isinstance(b, AbstractWeightedBlock) else None for b in self.blocks]

    @property
    def scalar_weight(self) -> torch.Tensor:
        return self.block.scalar_weight

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

    def delitem(self, idx: int) -> torch.Tensor:
        del self.blocks[idx]


class ModelFactory:
    def __init__(self, x: torch.Tensor, y: torch.Tensor, max_size: Optional[int] = None, epochs: int = 1000,
                 early_stopping: int = 5, lr: float = 1e-2, criterion_cls: nn.Module = nn.MSELoss) -> None:
        self.in_features = x.shape[1]
        self.out_features = y.shape[1]
        self.x = x
        self.y = y
        self.max_size = max_size
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.lr = lr
        self.criterion = criterion_cls()

    def fit_prune(self, model: Model) -> Model:
        if self.max_size is None:
            return model
        while len(model.blocks) > self.max_size:
            model = self.fit(model)
            model = self.prune(model)
        model = self.fit(model)
        return model

    def check_early_stopping(self, loss: float, loss_hist: List[float]) -> Tuple[bool, List[float]]:
        loss_hist.append(loss)
        if len(loss_hist) > self.early_stopping + 1:
            _ = loss_hist.pop(0)
            if loss >= max(loss_hist[:-1]):
                return True, loss_hist
        return False, loss_hist

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

    def prune(self, model: Model) -> Model:
        idx, _ = model.least_significant
        model.delitem(idx)
        return model

    def from_class_list(self, blocks_classes: List[Type[AbstractBlock]]) -> Model:
        model = Model(blocks=[bc(self.in_features, self.out_features) for bc in blocks_classes])
        model = self.fit_prune(model)
        return model

    def from_occurrence_dict(self, blocks_classes: Dict[Type[AbstractBlock], int]) -> Model:
        blocks = []
        for (cls, occurrence) in blocks_classes.items():
            blocks.extend(cls(self.in_features, self.out_features) for _ in range(occurrence))
        model = Model(blocks=blocks)
        model = self.fit_prune(model)
        return model


class MultiLayerModelFactory:
    def __init__(self, x: torch.Tensor, y: torch.Tensor, max_size: Optional[int] = None, layers: int = 1, epochs: int = 1000,
                 early_stopping: int = 5, lr: float = 1e-2, criterion_cls: nn.Module = nn.MSELoss) -> None:
        self.in_features = x.shape[1]
        self.out_features = y.shape[1]
        self.x = x
        self.y = y
        self.max_size = max_size
        self.layers = layers
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.lr = lr
        self.criterion = criterion_cls()

    def fit_prune(self, model: Model, full_model: Model = None) -> Model:
        if self.max_size is None:
            return model
        if full_model is None:
            full_model = model
        while len(model.blocks) > self.max_size:
            model = self.fit(full_model)
            model = self.prune(model)
        """
        for bc in model.blocks:
            print(model.blocks)
            if isinstance(bc, MultiLayerModel):
                print(bc)
                self.fit_prune(bc, full_model)
        """
        full_model = self.fit(full_model)
        return full_model

    def check_early_stopping(self, loss: float, loss_hist: List[float]) -> Tuple[bool, List[float]]:
        loss_hist.append(loss)
        if len(loss_hist) > self.early_stopping + 1:
            _ = loss_hist.pop(0)
            if loss >= max(loss_hist[:-1]):
                return True, loss_hist
        return False, loss_hist

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

    def prune(self, model: Model) -> Model:
        idx, _ = model.least_significant
        model.delitem(idx)
        return model

    def from_class_list(self, blocks_classes: List[Type[AbstractWeightedBlock]], block: Type[AbstractWeightedBlock] = LinearBlock,
                        current_layer: int = 1) -> Model:
        if self.layers == current_layer:
            model = MultiLayerModel(
                block=block(self.in_features, self.out_features),
                blocks=[bc(self.in_features, self.out_features) for bc in blocks_classes])
        else:
            model = MultiLayerModel(
                block=block(self.in_features, self.out_features),
                blocks=[self.from_class_list(blocks_classes, block=bc, current_layer = current_layer + 1) for bc in blocks_classes]
            )
        if current_layer == 1:
            model = self.fit_prune(model)
        return model

    def from_occurrence_dict(self, blocks_classes: Dict[Type[AbstractBlock], int]) -> Model:
        blocks = []
        for (cls, occurrence) in blocks_classes.items():
            blocks.extend(cls(self.in_features, self.out_features) for _ in range(occurrence))
        model = Model(blocks=blocks)
        model = self.fit_prune(model)
        return model
