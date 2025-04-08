import pandas as pd
from epoch_based_evolution import DynamicNN, create_dataloaders
import load_data
import ast

import torch
import torch.nn as nn
import torch.optim as optim

import time

ID_TO_TEST = 40996

# Just to extract the input and output sizes
X_train, y_train, X_val, y_val, X_test, y_test = load_data.get_preprocessed_data(dataset_id=ID_TO_TEST, scaling=True, random_seed=13, return_as='tensor')
input_size, output_size = load_data.get_tensor_sizes(X_train, y_train)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the results
results_raw = pd.read_csv('./Epoch-Based Evolution (EBE)/EBE-NAS_LCBench_results.csv')

dataset_df = results_raw[results_raw['dataset_id'] == ID_TO_TEST]
dataset_df = dataset_df[dataset_df['epoch'] == max(dataset_df['epoch'])].sort_values('val_acc')

N_MODELS = 5
best_models = dataset_df.head(N_MODELS)

reborn_models = {}
results = []
for index, model in best_models.iterrows(): 
    print('Testing model:', index + 1)
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
    train_loss, train_acc, val_loss, val_acc = reborn_models[index].es_train(train_loader, val_loader, es_patience=50, verbose=True)
    diff_time = time.time() - start_time

        # Save result
    results.append({
        'index': index,
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
