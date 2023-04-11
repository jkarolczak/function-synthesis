import torch

from blocks import CosBlock, SinBlock, Log2Block, LnBlock, Log10Block
from model import ModelFactory

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    x = torch.arange(1, 10, 0.1, dtype=torch.float32).unsqueeze(-1)
    y = torch.sin(x) + torch.log2(x)
    model = ModelFactory(x, y, max_size=2).from_class_list([CosBlock, SinBlock, Log2Block, LnBlock, Log10Block])
    print(torch.hstack([model.forward(x), y]))
    for b, w in zip(model.blocks, model.scalar_weights):
        print(b, w)
