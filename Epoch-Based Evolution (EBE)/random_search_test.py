import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from epoch_based_evolution import SearchSpace, DynamicNN, create_dataloaders
import load_data

import time
import pandas as pd

# Set IDs to test
dataset_ids = { 
    'Fashion': 40996,  # Fashion
    'Adult': 1590,   # adult
    'Higgs': 4532,   # higgs
    'Jasmine': 41143,  # jasmine
    'Vehicle': 54,     # vehicle
    'Volkert': 41166   # volkert
}

# Train for max time
dataset_max_times = {
    'Fashion': 1504,  # seconds
    'Adult': 1030,
    'Higgs': 2127,
    'Jasmine': 111,
    'Vehicle': 66,
    'Volkert': 1294
}


results = []

# Iterate across datasets
for i, (data_name, data_id) in enumerate(dataset_ids.items()):
    print(f'\nTesting for Dataset: {data_name} | {i+1}/{len(dataset_ids)}')
    
    # Load and preprocess data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data.get_preprocessed_data(
        dataset_id=data_id, scaling=True, random_seed=13, return_as='tensor')

    # Get input and output size
    input_size, output_size = load_data.get_tensor_sizes(X_train, y_train)

    # Establish SearchSpace
    search_space = SearchSpace(input_size=input_size, output_size=output_size)

    dataset_start_time = time.time()
    model_count = 0

    while True:
        elapsed_dataset_time = time.time() - dataset_start_time
        max_time = dataset_max_times[data_name]
        if elapsed_dataset_time >= max_time:
            print(f"Max time reached for dataset {data_name}. Trained {model_count} models.\n")
            break

        # Sample a new random architecture and model
        architecture = search_space.sample_architecture()
        batch_size = architecture['batch_size']
        model = search_space.create_model(architecture)

        # Get DataLoaders
        train_loader = create_dataloaders(X=X_train, y=y_train, batch_size=batch_size)
        val_loader = create_dataloaders(X=X_val, y=y_val, batch_size=batch_size)
        test_loader = create_dataloaders(X=X_test, y=y_test, batch_size=batch_size)

        # Train the model
        model_start_time = time.time()
        best_train_loss, best_train_acc, best_val_loss, best_val_acc = model.es_train(
            train_loader=train_loader,
            val_loader=val_loader,
            es_patience=50,
            max_epochs=1000,
            verbose=True
        )
        model_time = time.time() - model_start_time

        # Flatten architecture dictionary for easy dataframe integration
        flat_architecture = {f'arch_{k}': v for k, v in architecture.items()}

        # Collect results
        result_entry = {
            'dataset': data_name,
            'train_loss': best_train_loss,
            'train_acc': best_train_acc,
            'val_loss': best_val_loss,
            'val_acc': best_val_acc,
            'time_elapsed_sec': model_time,
            'total_time_elapsed_sec': elapsed_dataset_time + model_time,
            'model_index': model_count + 1
        }
        result_entry.update(flat_architecture)
        results.append(result_entry)

        model_count += 1

# Convert results to DataFrame and export
results_df = pd.DataFrame(results)
results_df.to_csv("random_search_results.csv", index=False)