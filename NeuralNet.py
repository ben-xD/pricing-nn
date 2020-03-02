from torch import nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, num_layers, layer_sizes, output_size):
        super(NeuralNet, self).__init__()

        print(layer_sizes)
        self.relu = nn.ReLU()
        self.linears = nn.ModuleList([nn.Linear(input_size, layer_sizes[0])])
        self.linears.extend([nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(0, num_layers-1)])
        self.linears.append(nn.Linear(layer_sizes[num_layers-1], output_size))
        print(self.linears)

    def forward(self, x):
        y = x
        for i in range(len(self.linears)):
          y = self.linears[i](y)
          #Apply relu to all layers but the last
          if(i < len(self.linears) - 1):
            y = self.relu(y)
        return y 
