import torch
import torch.nn as nn
import torch.optim as optim

# region NN Classes
# Base class for consistency
class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
    
    def forward(self, x):
        raise NotImplementedError("Each model must implement its own forward method.")

# Single Layer Perceptron
class SLP(BaseModel):
    def __init__(self, input_size, output_size, activation=None):
        super(SLP, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.activation = activation if activation else nn.Identity()
    
    def forward(self, x):
        return self.activation(self.fc(x))

# Radial Basis Function Network
class RBFN(BaseModel):
    def __init__(self, input_size, hidden_size, output_size, activation=None):
        super(RBFN, self).__init__()
        self.centers = nn.Parameter(torch.randn(hidden_size, input_size))
        self.beta = nn.Parameter(torch.ones(hidden_size) * 1.0)
        self.fc = nn.Linear(hidden_size, output_size)
        self.activation = activation if activation else nn.Identity()
    
    def forward(self, x):
        # Compute radial basis function activations
        dists = torch.cdist(x, self.centers)
        rbf_activations = torch.exp(-self.beta * (dists ** 2))
        return self.activation(self.fc(rbf_activations))

# Multilayer Perceptron
class MLP(BaseModel):
    def __init__(self, input_size, hidden_sizes, output_size, activation=nn.ReLU()):
        super(MLP, self).__init__()
        layers = []
        in_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(activation)
            in_size = hidden_size
        layers.append(nn.Linear(in_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Simple Recurrent Neural Network
class SimpleRNN(BaseModel):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, activation=None):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.activation = activation if activation else nn.Identity()
    
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.activation(self.fc(out[:, -1, :]))  # Use the last time step output

# Single Neuron Model
class SingleNeuron(BaseModel):
    def __init__(self, input_size, activation=None):
        super(SingleNeuron, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        self.activation = activation if activation else nn.Identity()
    
    def forward(self, x):
        return self.activation(self.fc(x))

# endregion