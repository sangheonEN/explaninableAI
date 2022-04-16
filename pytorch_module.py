import torch
import torch.nn as nn
import torch.nn.functional as F
import pathlib
from torch.utils.tensorboard import SummaryWriter

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(5, 10)
        self.fc2 = nn.Linear(10, 20)
        self.fc3 = nn.Linear(20, 30)
        self.fc4 = nn.Linear(30, 2)

        log_dir = pathlib.Path.cwd() / "tensorboard_logs"
        self.writer = SummaryWriter(log_dir)


    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        x = F.relu(x)

        return x


if __name__ == "__main__":
    x = torch.rand(1, 5)

    model = Model()
    y = model(x)

    print(y)


