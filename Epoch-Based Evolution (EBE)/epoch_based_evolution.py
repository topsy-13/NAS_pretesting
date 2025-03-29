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
    
    def evaluate(self, val_loader):
            self.eval()
            
            total = 0
            correct = 0
            running_loss = 0.0
            
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(self.device), labels.to(self.device)

                    # Automatically flatten if it's an image (i.e., has more than 2 dimensions)
                    if features.dim() > 2:  
                        features = features.view(features.size(0), -1)

                    outputs = self(features)
                    loss = self.criterion(outputs, labels)
                    running_loss += loss.item() * features.size(0)
                    
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_loss = running_loss / total
            val_accuracy = correct / total
            
            return val_loss, val_accuracy


class FlexibleDynamicNN(DynamicNN):
    def __init__(self, input_size, output_size, 
                 hidden_layers, 
                 activation_fns, dropout_rates,
                 lr, optimizer_type, 
                 batch_size, random_seed=None):
        # Call the parent constructor with default values
        super(FlexibleDynamicNN, self).__init__(input_size, output_size, 
                                                hidden_layers, 
                                                activation_fn=activation_fns[0], 
                                                dropout_rate=dropout_rates[0], 
                                                lr=lr, 
                                                optimizer_type=optimizer_type, 
                                                batch_size=batch_size, 
                                                random_seed=random_seed)
        
        layers = []
        prev_size = input_size

        # Create hidden layers dynamically with varying activation functions and dropout rates
        for i, size in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_size, size))
            
            # Use the activation function from the list
            if i < len(activation_fns):
                layers.append(activation_fns[i]())
            else:
                raise ValueError("Not enough activation functions provided for the hidden layers.")
            
            # Use the dropout rate from the list
            if i < len(dropout_rates) and dropout_rates[i] > 0:
                layers.append(nn.Dropout(dropout_rates[i]))
                
            prev_size = size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)
        self.optimizer = optimizer_type(self.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()


class SearchSpace():
    def __init__(self, input_size, output_size, 
                 min_layers=2, max_layers=7, 
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

    # def train_model(self, model, 
    #                 train_loader, val_loader, # because batch_size
    #                 epochs=1):
    #     train_loss, train_acc = model.oe_train(train_loader, num_epochs=epochs)

    #     return train_loss, train_acc    

# region Generations
class Generation():
    def __init__(self, search_space, n_individuals):
        self.search_space = search_space
        self.n_individuals = n_individuals
        self.generation = self.build_generation() 

    def build_generation(self):
        generation = {}
        for i in range(self.n_individuals):
            architecture = self.search_space.sample_architecture()
            model = self.search_space.create_model(architecture)
            generation[i] = {
                "model": model,
                "architecture": architecture,
            }
        return generation
    
    def train_generation(self, train_loader, num_epochs=1):
        for i in range(self.n_individuals):
            model = self.generation[i]["model"]
            train_loss, train_acc = model.oe_train(train_loader, num_epochs=num_epochs)
            self.generation[i]["train_loss"] = train_loss
            self.generation[i]["train_acc"] = train_acc
        return
    

    def validate_generation(self, val_loader):
        for i in range(self.n_individuals):
            model = self.generation[i]["model"]
            val_loss, val_acc = model.evaluate(val_loader)
            self.generation[i]["val_loss"] = val_loss
            self.generation[i]["val_acc"] = val_acc
        return


    def get_worst_individuals(self, 
                              percentile_drop=15):
    
        n_worst_individuals = max(1, int(self.n_individuals * percentile_drop / 100))  # Ensure at least 1

        # Sort individuals by validation loss in descending order (higher loss is worse)
        sorted_generation = sorted(self.generation.items(), key=lambda x: x[1]["val_loss"], reverse=True)

        # Extract the keys of the worst individuals
        self.worst_individuals = [key for key, _ in sorted_generation[:n_worst_individuals]]

        return

    def drop_worst_individuals(self):
        # Remove worst individuals
        for idx in self.worst_individuals:
            del self.generation[idx]
        
        # Re-index the remaining individuals to maintain continuous keys
        self.generation = {new_idx: val for new_idx, (_, val) in enumerate(self.generation.items())}
        self.n_individuals = len(self.generation)  # Update the count

    def drop_all_except_best(self):
        # Sort individuals by validation loss in ascending order (lower loss is better)
        sorted_generation = sorted(self.generation.items(), key=lambda x: x[1]["val_loss"])

        # Keep only the best individual
        best_individual = sorted_generation[0][0]
        self.generation = {0: self.generation[best_individual]}
        self.n_individuals = 1

    def train_best_individual(self, train_loader, num_epochs=1):
        best_model = self.generation[0]["model"]
        best_model.oe_train(train_loader, num_epochs=num_epochs)
        return

# region Functions
def run_generation(generation, train_loader, val_loader, num_epochs=1,
                   percentile_drop=15):
    
    # Generation is trained, and dropped
    generation.train_generation(train_loader, num_epochs=num_epochs)
    generation.validate_generation(val_loader)
    generation.get_worst_individuals(percentile_drop)
    generation.drop_worst_individuals()

    return generation
#endregion