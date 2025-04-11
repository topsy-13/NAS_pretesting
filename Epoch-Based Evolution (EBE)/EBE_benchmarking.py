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
    3, 6, 11, 12, 14, 15, 16, 18, 22, 23, 28, 29, 31, 32, 37, 38, 44, 46, 50, 54, 
    151, 182, 188, 300, 307, 458, 469, 554, 1049, 1050, 1053, 1063, 1067, 1068, 
    1461, 1462, 1464, 1468, 1475, 1478, 1480, 1485, 1486, 1487, 1489, 1494, 1497, 
    1501, 1510, 1590, 4134, 4534, 4538, 6332, 23381, 23517, 40499, 40668, 40670, 
    40701, 40923, 40927, 40966, 40975, 40978, 40979, 40982, 40983, 40984, 40994, 
    40996, 41027
]
results = [] 

csv_filename = "EBE-NAS_results.csv"

# Inicializar archivo con encabezado si no existe
if not os.path.exists(csv_filename):
    pd.DataFrame(columns=['dataset_id', 'epoch', 'model_id', 'val_acc', 'val_loss', 'num_models', 'epoch_time']).to_csv(csv_filename, index=False)

for i, data_id in enumerate(dataset_ids):
    print(f'Testing for Dataset {data_id} | {i+1}/{len(dataset_ids)}')
    X_train, y_train, X_val, y_val, X_test, y_test = load_data.get_preprocessed_data(
        dataset_id=data_id, scaling=True, random_seed=13, return_as='tensor')
    input_size, output_size = load_data.get_tensor_sizes(X_train, y_train)

    search_space = SearchSpace(input_size=input_size, output_size=output_size)

    N_INDIVIDUALS = 1000
    N_EPOCHS = 10
    percentile_drop = 25

    start_time = time.time()
    generation = Generation(search_space, N_INDIVIDUALS)
    results = []  # Reiniciar resultados por cada dataset

    for n_epoch in range(N_EPOCHS + 1):
        epoch_start_time = time.time()
        print(f'\n-Epoch: {n_epoch}')
        final_gen = run_generation(generation, X_train, y_train, X_val, y_val, percentile_drop=percentile_drop)
        num_models = len(final_gen.generation)
        epoch_time = time.time() - epoch_start_time
        
        if num_models <= 1:
            print(f"No models left to evaluate at epoch {n_epoch}. Stopping early.")
            break

        for model_id, model_data in final_gen.generation.items():
            results.append({
                'dataset_id': data_id,
                'epoch': n_epoch,
                'model_id': model_id,
                'val_acc': model_data['val_acc'],
                'val_loss': model_data['val_loss'],
                'num_models': num_models,
                'epoch_time': epoch_time
            })
        
        print(f"Survivor models: {num_models}")
        percentile_drop += 2  # Incremento por cada epoch
    
    final_time = time.time() - start_time
    print(f"Dataset {data_id} completed in {final_time:.2f} seconds")

    # Guardar resultados después de cada dataset
    df = pd.DataFrame(results)
    df.to_csv(csv_filename, mode='a', header=False, index=False)

print(f"Results saved to {csv_filename}")


