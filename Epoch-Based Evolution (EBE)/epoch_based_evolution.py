import torch
import torch.nn as nn
import torch.optim as optim

import random

# Definir una arquitectura de red flexible 
class DynamicNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, activation_fn, dropout_rate):
        super(DynamicNN, self).__init__()
        
        layers = []
        prev_size = input_size

        # Crear capas ocultas dinámicamente
        for size in hidden_layers:
            layers.append(nn.Linear(prev_size, size))
            layers.append(activation_fn())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = size
        
        # Capa de salida
        layers.append(nn.Linear(prev_size, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def sample_architecture(input_size, output_size):
    hidden_layers = random.choices([16, 32, 64, 128], k=random.randint(2, 5))
    activation_fn = random.choice([nn.ReLU, nn.LeakyReLU, nn.Sigmoid])
    dropout_rate = random.choice([0, 0.2, 0.5])
    optimizer_type = random.choice([optim.Adam, optim.SGD])
    learning_rate = random.uniform(0.001, 0.01)

    return {
        "hidden_layers": hidden_layers,
        "activation_fn": activation_fn,
        "dropout_rate": dropout_rate,
        "optimizer_type": optimizer_type,
        "learning_rate": learning_rate
    }


# 