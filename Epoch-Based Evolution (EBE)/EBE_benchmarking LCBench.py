import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from epoch_based_evolution import SearchSpace, Generation, run_generation
import load_data

import time
import pandas as pd
import os

# Set IDs to test
dataset_ids = [ 
    40996,  # Fashion
    1590,   # adult
    4532,   # higgs
    41143,  # jasmine
    54,     # vehicle
    41166   # volkert
]

csv_filename = "EBE-NAS_LCBench_results.csv"

# Inicializar archivo si no existe
if not os.path.exists(csv_filename):
    pd.DataFrame(columns=[
        'hidden_layers', 'activation_fn', 'dropout_rate', 'optimizer_type',
        'learning_rate', 'batch_size', 'random_seed', 'train_loss', 'train_acc',
        'val_loss', 'val_acc', 'dataset_id', 'epoch', 'epoch_time'
    ]).to_csv(csv_filename, index=False)

for i, data_id in enumerate(dataset_ids):
    print(f'\nTesting for Dataset {data_id} | {i+1}/{len(dataset_ids)}')
    X_train, y_train, X_val, y_val, X_test, y_test = load_data.get_preprocessed_data(
        dataset_id=data_id, scaling=True, random_seed=13, return_as='tensor')
    input_size, output_size = load_data.get_tensor_sizes(X_train, y_train)

    search_space = SearchSpace(input_size=input_size, output_size=output_size)

    N_INDIVIDUALS = 500
    N_EPOCHS = 5
    percentile_drop = 25

    start_time = time.time()
    generation = Generation(search_space, N_INDIVIDUALS)

    all_results = []  

    for n_epoch in range(N_EPOCHS + 1):
        epoch_start_time = time.time()
        print(f'\n-Epoch: {n_epoch}')
        final_gen = run_generation(generation, X_train, y_train, X_val, y_val, percentile_drop=percentile_drop)
        num_models = len(final_gen.generation)
        epoch_time = time.time() - epoch_start_time

        results_df = final_gen.return_df()
        results_df['dataset_id'] = data_id
        results_df['epoch'] = n_epoch
        results_df['epoch_time'] = epoch_time

        all_results.append(results_df)  

        if num_models <= 1:
            print(f"No models left to evaluate at epoch {n_epoch}. Stopping early.")
            break

        print(f"Survivor models: {num_models}")
        percentile_drop += 5

    final_time = time.time() - start_time
    log_message = f"Dataset {data_id} completed in {final_time:.2f} seconds\n"

    with open("process_log.txt", "a") as log_file:
        log_file.write(log_message)

    # Concatenar todos los resultados del dataset y guardar en el CSV
    if all_results:
        pd.concat(all_results, ignore_index=True).to_csv(csv_filename, mode='a', header=False, index=False)

print(f"All results saved to {csv_filename}")



