import experiments
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import combinations_with_replacement, product
import os

# Own modules
import load_cifar10
import experiments

import pandas as pd
import numpy as np

# Load the datasets
train_dataset, val_dataset, test_dataset = load_cifar10.load_and_create_loaders('./CIFAR-10', return_ds=True)

# Set constant architecture
input_size = 32 * 32 * 3  # 3072 features per image
num_classes = 10  # CIFAR-10 has 10 classes

# Set search space
n_hidden_layers = [1, 2, 4]
n_neurons_x_layer = [50, 200, 1000]
learning_rate = [10**-3, 10**-4, 10**-5]


architectures = experiments.generate_architectures(n_hidden_layers,
                                                    n_neurons_x_layer,
                                                    learning_rate, 
                                                    input_size, num_classes,
                                                    symmetric=True) # For symmetric MLP:

# Set other HP
random_seed = 13
batch_size = 32

results_list = []

for i, architecture in enumerate(architectures):
    print(f'Training architecture {i+1} / {len(architectures)}')
    experiment = experiments.Experiment(architecture=architecture, 
                                    train_strategy='ES', 
                                    random_seed=random_seed)
    results = experiment.full_experiment(train_dataset=train_dataset,
                                     val_dataset=val_dataset, 
                                     test_dataset=test_dataset,
                                     batch_size=batch_size, verbose=True)
    
    string_values = {key: str(value) for key, value in results.items()}
    results_list.append(string_values)


EXPORT_NAME = 'Classic_27' 
results_df = pd.DataFrame(results_list)
results_df.to_csv(f'./OE-Tests/First_Experiments/{EXPORT_NAME}ES.csv', index=False)