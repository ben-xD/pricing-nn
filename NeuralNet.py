from torch import nn

class NeuralNet(nn.Module):



    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        # self.fc1 = nn.Linear(input_size, output_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.norm = nn.BatchNorm1d(num_features=hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.norm(out)
        out = self.relu(out)
        out = self.fc2(out)
        # out = self.sigmoid(out)
        return out

    def accuracy(self, out, yb, batch_size):
        correct = 0
        sigmoid = nn.Sigmoid()
        preds = sigmoid(out).round()
        for i in range(batch_size):
            if (preds[i] == yb[i]):
                correct += 1

        # set_trace()
        return correct / batch_size