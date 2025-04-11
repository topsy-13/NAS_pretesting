import pandas as pd
from epoch_based_evolution import DynamicNN, create_dataloaders
import load_data
import ast

import torch
import torch.nn as nn
import torch.optim as optim

import time

# Set IDs to test
dataset_ids = { 
    'Fashion': 40996,  # Fashion
    'Adult': 1590,   # adult
    'Higgs': 4532,   # higgs
    'Jasmine': 41143,  # jasmine
    'Vehicle': 54,     # vehicle
    'Volkert': 41166   # volkert
}

# Load the results
results_raw = pd.read_csv('./Epoch-Based Evolution (EBE)/EBE-NAS_LCBench_results.csv')

results = []

for data_name, data_id in dataset_ids.items():
    print(f'\n **Testing for Dataset: {data_name}**')
    # Just to extract the input and output sizes
    X_train, y_train, X_val, y_val, X_test, y_test = load_data.get_preprocessed_data(dataset_id=data_id, scaling=True, random_seed=13, return_as='tensor')
    input_size, output_size = load_data.get_tensor_sizes(X_train, y_train)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    dataset_df = results_raw[results_raw['dataset_id'] == data_id]
    dataset_df = dataset_df[dataset_df['epoch'] == max(dataset_df['epoch'])].sort_values('val_acc', ascending=False)

    n_models_to_test = {
        "Adult": 15,
        "Fashion": 11,
        "Higgs": 13, # NAN VALUES PRESENTED
        "Jasmine": 34,
        "Vehicle": 41,
        "Volkert": 8
    }

    N_MODELS = n_models_to_test[data_name]
    best_models = dataset_df.head(N_MODELS).reset_index(drop=True)

    reborn_models = {}
    for index, model in best_models.iterrows(): 
        print('--Testing model:', index + 1, '/', N_MODELS)
        # Extract attributes
        hidden_layers = ast.literal_eval(model['hidden_layers'])
        activation_fn_str = model['activation_fn'].split("'")[1].split('.')[-1]
        if activation_fn_str == 'Sigmoid':
            activation_fn = nn.Sigmoid
        elif activation_fn_str == 'ReLU':
            activation_fn = nn.ReLU
        elif activation_fn_str == 'Tanh':
            activation_fn = nn.Tanh
        elif activation_fn_str == 'LeakyReLU':
            activation_fn = nn.LeakyReLU

        dropout_rate = model['dropout_rate']
        lr = model['learning_rate']
        batch_size = model['batch_size']
        optimizer_str = model['optimizer_type'].split("'")[1].split('.')[-1]
        if optimizer_str == 'Adam':
            optimizer_type = optim.Adam
        elif optimizer_str == 'SGD':
            optimizer_type = optim.SGD

        # Architectures are reborn
        reborn_models[index] = DynamicNN(input_size, output_size, 
                    hidden_layers, 
                    activation_fn, dropout_rate,
                    lr, optimizer_type).to(device)
        
        train_loader = create_dataloaders(X=X_train, y=y_train, batch_size=batch_size)
        val_loader = create_dataloaders(X=X_val, y=y_val, batch_size=batch_size)
        
        # Training time
        start_time = time.time()
        train_loss, train_acc, val_loss, val_acc = reborn_models[index].es_train(train_loader, val_loader, es_patience=50, verbose=True, max_epochs=1000)
        diff_time = time.time() - start_time

        # Append results
        results.append({
            'index': index,
            'dataset_id': data_name,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'training_time_sec': diff_time,
            'hidden_layers': hidden_layers,
            'activation_fn': activation_fn.__class__.__name__,
            'dropout_rate': dropout_rate,
            'learning_rate': lr,
            'batch_size': batch_size,
            'optimizer_type': optimizer_type.__name__,
        })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
results_df.to_csv('./Epoch-Based Evolution (EBE)/models_ES_training_results.csv', index=False)
