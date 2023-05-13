import torch

from blocks import CosBlock, SinBlock, LinearBlock
from model import MultiLayerModelFactory

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    x = torch.arange(1, 20, 0.1, dtype=torch.float32).unsqueeze(-1)
    y = torch.sin(2 * x + torch.sin(x)) + torch.cos(torch.cos(x)) + 0.1 * x
    model = MultiLayerModelFactory(x, y, max_size=2, layers=2, epochs=1000).from_class_list(
        [CosBlock, SinBlock, LinearBlock])
    print(torch.hstack([model.forward(x), y]))
    print(model)
