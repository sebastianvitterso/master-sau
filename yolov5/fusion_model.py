
import torch
from models.master.model import Model


if __name__ == '__main__':
    model = Model()

    x1 = torch.zeros(1, 3, 256, 256)
    x2 = torch.zeros(1, 3, 256, 256)
    y = model(x1, x2)

    print(y)