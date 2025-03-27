import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np

def set_seed(seed=13):
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU (if available)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disables auto-optimization for conv layers (useful for exact reproducibility)
    return


# Definir una arquitectura de red flexible 
class DynamicNN(nn.Module):
    def __init__(self, input_size, output_size, 
                 hidden_layers, 
                 activation_fn, dropout_rate,
                 lr, optimizer_type, 
                 batch_size, random_seed # should I pass it?
                 ):
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

        self.network = nn.Sequential(*layers)
        self.optimizer = optimizer_type(self.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return
    

    def forward(self, x):
        return self.network(x)

    def oe_train(self, train_loader, num_epochs=1):
        self.train()
        
        for epoch in range(num_epochs):
            total = 0
            correct = 0
            running_loss = 0.0
            
            for features, labels in train_loader:
                features, labels = features.to(self.device), labels.to(self.device)

                # Automatically flatten if it's an image (i.e., has more than 2 dimensions)
                if features.dim() > 2:  
                    features = features.view(features.size(0), -1)  # Flatten images

                self.optimizer.zero_grad()
                outputs = self(features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # Accumulate loss
                running_loss += loss.item() * features.size(0)

                # Compute accuracy
                with torch.no_grad():
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            # Compute final loss and accuracy for the epoch
            train_loss = running_loss / total
            train_acc = correct / total
            
        return train_loss, train_acc




class SearchSpace():
    def __init__(self, input_size, output_size, 
                 min_layers=2, max_layers=10, 
                 min_neurons=13, max_neurons=512,
                 activation_fns=[nn.ReLU, nn.LeakyReLU, nn.Sigmoid],
                 dropout_rates=[0, 0.1, 0.2, 0.5],
                 min_learning_rate=0.001, max_learning_rate=0.01,
                 random_seeds=[13, 42, 1337, 2024, 777],
                 min_batch_size=128, max_batch_size=1024):
        
        self.input_size = input_size
        self.output_size = output_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layers = [min_layers, max_layers]
        self.neurons = [min_neurons, max_neurons]
        self.activation_fns = activation_fns
        self.learning_rates = [min_learning_rate, max_learning_rate]
        self.dropout_rates = dropout_rates
        self.optimizers = [optim.Adam, optim.SGD]

        self.random_seeds = random_seeds
        # Build batch sizes considering powers of 2
        power = 1
        self.batch_sizes = []
        while power <= max_batch_size:
            if power >= min_batch_size:
                self.batch_sizes.append(power)
            power *= 2


    def sample_architecture(self):
        hidden_layers = random.choices(range(self.neurons[0], self.neurons[1]), 
                                       k=random.randint(self.layers[0], self.layers[1]))
        activation_fn = random.choice(self.activation_fns)
        dropout_rate = random.choice(self.dropout_rates)
        optimizer_type = random.choice(self.optimizers)
        learning_rate = random.uniform(self.learning_rates[0], self.learning_rates[1])
        batch_size = random.choice(self.batch_sizes) # for later
        random_seed = random.choice(self.random_seeds)

        return {
            "hidden_layers": hidden_layers,
            "activation_fn": activation_fn,
            "dropout_rate": dropout_rate,
            "optimizer_type": optimizer_type,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "random_seed": random_seed
        }

    def create_model(self, architecture):
        hidden_layers = architecture["hidden_layers"]
        activation_fn = architecture["activation_fn"]
        dropout_rate = architecture["dropout_rate"]
        optimizer_type = architecture["optimizer_type"]
        learning_rate = architecture["learning_rate"]
        batch_size = architecture["batch_size"]  
        random_seed = architecture["random_seed"]

        # Create model
        model = DynamicNN(self.input_size, self.output_size, 
                          hidden_layers, activation_fn, 
                          dropout_rate, learning_rate, optimizer_type, 
                          batch_size, random_seed).to(self.device)

        return model

    def train_model(self, model, 
                    train_loader, val_loader, # because batch_size
                    num_epochs=1):
        train_loss, train_acc = model.oe_train(train_loader, num_epochs=num_epochs)

        return train_loss, train_acc    

# region Functions

#endregion