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
# Set other HP
seeds = [13, 42, 1337, 2024, 777]
batch_sizes = [128, 256, 512]

pairing_number = 1 
total_combinations = len(seeds) * len(batch_sizes)

# Symmetrical
architectures = experiments.generate_architectures(n_hidden_layers,
                                                    n_neurons_x_layer,
                                                    learning_rate, 
                                                    input_size, num_classes,
                                                    symmetric=True) # For symmetric MLP
for random_seed in seeds:
    for batch_size in batch_sizes:
        print(f'Testing seed-batch pairing number: {pairing_number}/{total_combinations}')

        # # For OE
        experiments.run_architecture_experiments(architectures=architectures,
                                                train_dataset=train_dataset, 
                                                val_dataset=val_dataset,
                                                test_dataset=test_dataset, 
                                                batch_size=batch_size,
                                                random_seed=random_seed,
                                                train_strategy='OE',
                                                verbose=True,
                                                export_path=f'Experiment Results/Symmetrical MLP/Seeds_Batches/OE_Initial27_{random_seed}-{batch_size}')


# # Asymmetrical
# architectures = experiments.generate_architectures(n_hidden_layers,
#                                                     n_neurons_x_layer,
#                                                     learning_rate, 
#                                                     input_size, num_classes,
#                                                     symmetric=False)
# # For OE
# experiments.run_architecture_experiments(architectures=architectures,
#                                          train_dataset=train_dataset, 
#                                          val_dataset=val_dataset,
#                                          test_dataset=test_dataset, 
#                                          batch_size=batch_size,
#                                          random_seed=random_seed,
#                                          train_strategy='OE',
#                                          export_path=f'Experiment Results/Asymmetrical MLP/OE_Initial{len(architectures)}')

# For ES
# experiments.run_architecture_experiments(architectures=architectures,
#                                          train_dataset=train_dataset, 
#                                          val_dataset=val_dataset,
#                                          test_dataset=test_dataset, 
#                                          batch_size=batch_size,
#                                          random_seed=random_seed,
#                                          train_strategy='ES',
#                                          export_path='Experiment Results/Symmetrical MLP/ES_Initial27.csv')




os.system("shutdown /s /t 60")  # Shutdown in 60 seconds