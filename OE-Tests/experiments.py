import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset

import numpy as np
import random
import time
import copy
from itertools import product

import pandas as pd

def set_seed(seed=13):
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU (if available)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disables auto-optimization for conv layers (useful for exact reproducibility)
    return


class MLP(nn.Module):
    def __init__(self, input_size, neuron_structure:list, num_classes, hp:dict):
        super().__init__()
        # Define first layer
        layers = []
        prev_size = input_size
        # Create hidden layers
        # Neuron Structure is expecting a list
        for neurons in neuron_structure:
            layers.append(nn.Linear(prev_size, neurons))
            layers.append(nn.ReLU())
            prev_size = neurons

        # Define output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Define optimizer and loss function
        self.optimizer = optim.Adam(self.parameters(), lr=hp['lr'])
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        return
    

    def forward(self, x):
        return self.network(x)
    
    def _compute_metrics(self, loader):
        """Compute loss and accuracy for a given dataset loader."""
        total = 0
        correct = 0
        running_loss = 0.0

        with torch.no_grad():  # No gradient calculation
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)

                images = images.view(images.size(0), -1)  # Flatten images

                outputs = self(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                predicted = torch.argmax(outputs, dim=1)  # More efficient than `torch.max`
                total += labels.size(0)
                correct += torch.count_nonzero(predicted == labels).item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        return epoch_loss, epoch_acc


    def oe_train(self, train_loader):
        self.train()
        total = 0
        correct = 0
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            # Flatten images from (batch, 3, 32, 32) to (batch, 3072)
            images = images.view(images.size(0), -1)

            self.optimizer.zero_grad()
            outputs = self(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # Metrics
            # Accumulate loss (convert from mean loss to total loss)
            running_loss += loss.item() * images.size(0)
            # Compute accuracy without tracking gradients
            with torch.no_grad():
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

         # Compute final loss and accuracy
        train_loss = running_loss / total
        train_acc = correct / total
        
        return train_loss, train_acc

    
    def es_train(self, train_loader, val_loader, es_patience=50, max_epochs=500, verbose=False):
        # Early stopping parameters
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        best_model_state = None  

        learning_curve = {
            "epoch": [],
            "LC_Train_Loss": [],
            "LC_Train_Acc": [],
            "LC_Val_Loss": [],
            "LC_Val_Acc": []
        }

        for epoch in range(1, max_epochs + 1):  # Avoids infinite loop
            epoch_train_loss, epoch_train_acc = self.oe_train(train_loader)

            # Validation set (no gradients tracked)
            with torch.no_grad():
                epoch_val_loss, epoch_val_acc = self.evaluate(val_loader)

            # Record learning curve
            learning_curve["epoch"].append(epoch)
            learning_curve["LC_Train_Loss"].append(epoch_train_loss)
            learning_curve["LC_Train_Acc"].append(epoch_train_acc)
            learning_curve["LC_Val_Loss"].append(epoch_val_loss)
            learning_curve["LC_Val_Acc"].append(epoch_val_acc)

            # Check for improvement
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                epochs_without_improvement = 0
                best_model_state = copy.deepcopy(self.state_dict())  # Deep copy the model state
            else:
                epochs_without_improvement += 1

            # Check for early stopping
            if epochs_without_improvement >= es_patience:
                if verbose:
                    print(f"Early stopping triggered after {epoch} epochs.")
                break

        # Restore best model before returning
        if best_model_state:
            self.load_state_dict(best_model_state)

        # Retrieve the best recorded metrics (from `learning_curve`)
        best_index = learning_curve["LC_Val_Loss"].index(best_val_loss)
        best_train_loss = learning_curve["LC_Train_Loss"][best_index]
        best_train_acc = learning_curve["LC_Train_Acc"][best_index]
        best_val_acc = learning_curve["LC_Val_Acc"][best_index]

        return best_train_loss, best_train_acc, best_val_loss, best_val_acc, learning_curve
    
    
    def evaluate(self, test_loader):
        """Evaluate the model without updating weights."""
        self.eval()  # Set model to evaluation mode
        eval_loss, eval_acc = self._compute_metrics(test_loader)
        self.train()  # Restore training mode
        return eval_loss, eval_acc
    
    
class Experiment:

    def __init__(self, architecture:dict, train_strategy:str='OE', random_seed:int=13):
        """_summary_

        Args:
        architecture (dict): A dictionary defining the architecture of the model.
            Keys:
                "input_size" (int): The size of the input layer.
                "hlayers_size" (list): A list of integers representing the sizes of the hidden layers, the amount of layers is inferred based on the length of the list.
                "lr" (float): The Learning Rate.
                "num_classes" (int): The len of the last layer

        train_strategy (str, optional): The training strategy to use. Either 'OneEpoch' ('OE') or 'EarlyStopping' ('ES'). Defaults to 'OE'.
        random_seed (int, optional): A seed for random number generation to ensure reproducibility. Defaults to 13.


        Raises:
            ValueError: _description_
        """
        # Get architecture params
        hidden_layers_size = architecture['hlayers_size']
        n_hidden_layers = len(hidden_layers_size)
        lr = architecture['lr']
        

        self.architecture = {
            'input_size': architecture['input_size'],
            'hlayers_size': hidden_layers_size, 
            'n_hlayers': n_hidden_layers, 
            'lr': lr,
            'output_size': architecture['num_classes']

        }
        self.id = str(self.architecture['n_hlayers']) + '_' +  str(self.architecture['hlayers_size']) + '_' + str(self.architecture['lr'])

        self.strategy = train_strategy
        self.random_seed = random_seed
        
        if train_strategy not in ['OE', 'ES']:
            raise ValueError(f"Invalid strategy '{train_strategy}'. Valid options are: ['OE', 'ES'].")
        pass
    
            
    
    def build_MLP(self):
        set_seed(self.random_seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MLP(input_size=self.architecture['input_size'], 
                         neuron_structure=self.architecture['hlayers_size'],
                         num_classes=self.architecture['output_size'],
                         hp={'lr': self.architecture['lr']}
                         ).to(self.device)
        return
    

    def train_and_evaluate(self, train_loader, val_loader, test_loader=None):
        
        # Initialize neuron data
        results = {
            'ID': self.id,
            'n_layers': self.architecture['n_hlayers'],
            'neurons_per_layer': self.architecture['hlayers_size'],
            'learning_rate': self.architecture['lr'],
        }

        # Initialize variables for losses and accuracies
        train_loss, train_acc = None, None
        val_loss, val_acc = None, None
        test_loss, test_acc = None, None

        # ? Where to start the timer?
        
        # Get loss, accuracy per dataset
        if self.strategy == 'OE':
            # Start timing
            start_time = time.time()
            train_loss, train_acc = self.model.oe_train(train_loader)

            val_loss, val_acc = self.model.evaluate(val_loader)
            epoch_time_diff = time.time() - start_time # * After validation because training for ES it considers validation time as well.
            print('End train_Val OE time:', epoch_time_diff)
            
            if test_loader is not None:
                test_loss, test_acc = self.model.evaluate(test_loader)

        elif self.strategy == 'ES':
            # Start timing
            start_time = time.time()
            train_loss, train_acc, val_loss, val_acc, lc = self.model.es_train(train_loader=train_loader, val_loader=val_loader, es_patience=50)
            epoch_time_diff = time.time() - start_time

            results.update(lc)
            if test_loader is not None:
                test_loss, test_acc = self.model.evaluate(test_loader)

        # Store results
        results.update({
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'train_val_time': epoch_time_diff
        })
        return results


    # def generate_learning_curve(self, train_dataset, val_dataset, batch_size=32, metric='loss', verbose=False):
    #     """
    #     Generate a learning curve based on the dataset size using train and validation datasets.
    #     """

    #     # Get the full size of the training dataset
    #     full_train_size = len(train_dataset)
        
    #     # Create test loader (consistent for all evaluations)
    #     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
    #     DATASET_SIZES = 20
    #     training_sizes = np.logspace(1, np.log10(full_train_size), DATASET_SIZES, dtype=int)

    #     # Lists to store results
    #     train_accuracies = []
    #     val_accuracies = []
    #     train_losses = []
    #     val_losses = []
        
    #     # For each training size
    #     for train_size in training_sizes:
    #         if verbose:
    #             print(f"\nTraining with {train_size} samples...")
            
    #         # Get random subset of the training data
    #         indices = torch.randperm(full_train_size)[:train_size]
    #         train_subset = Subset(train_dataset, indices)
            
    #         # Create train loader from subset
    #         train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            
    #         # Train and evaluate model
    #         self.build_MLP()
    #         results = self.train_and_evaluate(train_loader=train_loader, val_loader=val_loader)

    #         # Store results
    #         train_accuracies.append(results['train_accuracy'])
    #         train_losses.append(results['train_loss'])
    #         val_accuracies.append(results['val_accuracy'])
    #         val_losses.append(results['val_loss'])
            
    #     # lc_dict = {
    #     #     'Dataset Size': training_sizes,
    #     #     'Train Accuracy': train_accuracies,
    #     #     'Train Losses': train_losses,
    #     #     'Val Accuracy': val_accuracies,
    #     #     'Val Losses': val_losses,
    #     # }
        
    #     if metric == 'loss':
    #         return train_losses, val_losses
    #     elif metric == 'acc':
    #         return train_accuracies, val_accuracies
    #     else:
    #         raise ValueError(f"Invalid metric'{metric}'. Valid options are: ['loss', 'acc'].")

    def generate_learning_curve(self, train_dataset, val_dataset, 
                                batch_size, verbose=False, epochs=30):
        
        # Load data
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        # Lists to store results
        train_accuracies = []
        val_accuracies = []
        train_losses = []
        val_losses = []

        self.build_MLP()
        if self.strategy == 'OE':
            for epoch in range(epochs):
                if verbose:
                    print(f'Epoch {epoch + 1} / {epochs}')
                
                # Train and evaluate model for one epoch
                results = self.train_and_evaluate(train_loader=train_loader,
                                                val_loader=val_loader)
                # Store results
                train_accuracies.append(results['train_accuracy'])
                train_losses.append(results['train_loss'])
                val_accuracies.append(results['val_accuracy'])
                val_losses.append(results['val_loss'])

            n_epochs = epochs
        else:
            # Train and evaluate model for ES
            results = self.train_and_evaluate(train_loader=train_loader,
                                              val_loader=val_loader)
            train_accuracies = results['train_acc']
            train_losses = results['train_loss']
            val_accuracies = results['val_acc']
            val_losses = results['val_loss']
            n_epochs = results['epoch']

        return train_accuracies, train_losses, val_accuracies, val_losses


    def full_experiment(self, train_dataset, val_dataset, test_dataset, batch_size, verbose=False):
        self.batch_size = batch_size

        set_seed(self.random_seed)
        # Build the loader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        
        # Build, train, validate and test the model
        self.build_MLP()
        if self.strategy == 'OE':
            results = self.train_and_evaluate(train_loader=train_loader,
                                            val_loader=val_loader, 
                                            test_loader=test_loader)
            
            # Get the learning curve of the model, it gets redone inside the function
            train_accuracies, train_losses, val_accuracies, val_losses = self.generate_learning_curve(train_dataset=train_dataset, 
                                        val_dataset=val_dataset, 
                                        batch_size=batch_size, verbose=verbose)
            # return everything as a dict
            results['LC_Train_Loss'] = train_losses
            results['LC_Val_Loss'] = val_losses
            results['LC_Train_Acc'] = train_accuracies
            results['LC_Val_Acc'] = val_accuracies
        else: 
            results = self.train_and_evaluate(train_loader=train_loader,
                                            val_loader=val_loader, 
                                            test_loader=test_loader)
            
        results['Strategy'] = self.strategy
        results['Seed'] = self.random_seed
        results['Batch Size'] = self.batch_size

        return results


def generate_architectures(n_hidden_layers, n_neurons_x_layer, learning_rate, input_size, num_classes, symmetric=False):

    architectures = []

    # For symmetric MLP:
    if symmetric: 
        search_space = list(product(n_hidden_layers, n_neurons_x_layer, learning_rate))

        for i, (h, n, lr) in enumerate(search_space):
                # Set layers w same size
                neuron_structure = (np.ones(h) * n).astype(int)

                architecture = {
                                'input_size': input_size,
                                'hlayers_size': neuron_structure,
                                'lr': lr,
                                'num_classes': num_classes
                                }
                architectures.append(architecture)

    else:
        # For each number of hidden layers
        for h in n_hidden_layers:
            # Generate all possible neuron configurations for h layers
            neuron_configs = product(n_neurons_x_layer, repeat=h)
            
            # For each neuron configuration and learning rate
            for neuron_config in neuron_configs:
                for lr in learning_rate:
                    # Add this combination to the search space
                    architecture = {
                        'input_size': input_size,
                        'hlayers_size': neuron_config,
                        'lr': lr,
                        'num_classes': num_classes
                    }
                    architectures.append(architecture)
        
    print('Total of architectures:', len(architectures))
    return architectures


def run_architecture_experiments(architectures, train_dataset, val_dataset, test_dataset, batch_size, random_seed,
                                export_path, 
                                train_strategy, verbose=False):

    results_list = []
    
    for i, architecture in enumerate(architectures):
        print(f'Training architecture {i+1} / {len(architectures)}')
        experiment = Experiment(architecture=architecture, train_strategy=train_strategy, random_seed=random_seed)
        results = experiment.full_experiment(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=batch_size, 
            verbose=verbose
        )
        
        string_values = {key: str(value) for key, value in results.items()}
        results_list.append(string_values)
    
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(f'./OE-Tests/{export_path}.csv', index=False)
    
