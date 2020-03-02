from torch import nn

class NeuralNet(nn.Module):
    def _init_(self, input_size, output_size):
        super(NeuralNet, self)._init_()

        self.fc1 = nn.Linear(input_size, 4)
        self.hidden1 = nn.Linear(4, 22)
        self.hidden2 = nn.Linear(22, 26)
        self.hidden3 = nn.Linear(26, 64)
        self.fc2 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.hidden1(out)
        out = self.relu(out)
        out = self.hidden2(out)
        out = self.relu(out)
        out = self.hidden3(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out
