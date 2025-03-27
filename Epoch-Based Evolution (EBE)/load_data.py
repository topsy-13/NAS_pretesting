import openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# region Loading data
def load_openml_dataset(dataset_id=334, scaling=True, random_seed=None, return_as='tensor'):
    # Get from OpenML
    dataset = openml.datasets.get_dataset(dataset_id)
    # Make it a df
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    
    # Convert to numpy array
    X = X.values
    
    # Ensure y is in a compatible format
    if isinstance(y, pd.Series):
        y = y.astype(str).values  # Convert categorical to string and then to numpy array
    
    # Apply LabelEncoder if y is not numeric
    if not np.issubdtype(y.dtype, np.number):
        print("Class column is not numeric. Applying LabelEncoder.")
        y = LabelEncoder().fit_transform(y)
    
    # Split the data
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=random_seed
    )
    
    # Scale the data 
    if scaling:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    
    if return_as == 'tensor':
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_val = torch.tensor(y_val, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)


    print(f'Data loaded successfully! as {type(y_train)}')
    print(f'Training data shape: {X_train.shape}')
    return X_train, y_train, X_val, y_val, X_test, y_test
# Use case:
# X_train, y_train, X_val, y_val, X_test, y_test = load_openml_dataset(return_as='tensor')

def create_dataset_and_loader(X, y, batch_size, shuffle=True):
    """
    Creates a TensorDataset from X and y tensors and wraps it in a DataLoader.
    """
    # Create the TensorDataset
    dataset = TensorDataset(X, y)
    # Create the DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataset, loader


def get_tensor_sizes(X_train, y_train):
    """
    Determine input and output sizes for PyTorch tensors
    
    Parameters:
    -----------
    X_train : torch.Tensor
        Input features tensor
    y_train : torch.Tensor
        Labels/target tensor
    
    Returns:
    --------
    tuple: (input_size, output_size)
    """
    # Input size: number of features (second dimension of input tensor)
    if len(X_train.shape) == 2:
        input_size = X_train.shape[1]
    elif len(X_train.shape) == 1:
        input_size = 1
    else:
        # For multi-dimensional inputs like images
        input_size = X_train.shape[1] * X_train.shape[2] * X_train.shape[3]

    # Output size: number of unique classes for classification
    # For regression, output size will be 1
    if y_train.dim() == 1:
        # Classification
        output_size = len(torch.unique(y_train))
    else:
        # Regression or multi-output
        output_size = y_train.shape[1]

    return input_size, output_size


# endregion

# Use case:
# batch_size = 512
# dataset_id=334

# X_train, y_train, X_val, y_val, X_test, y_test = load_openml_dataset(return_as='tensor', dataset_id=dataset_id, scaling=True)

# train_dataset, train_loader = create_dataset_and_loader(X_train, y_train,
#                                                         batch_size=batch_size)
# val_dataset, val_loader = create_dataset_and_loader(X_val, y_val, 
#                                                     batch_size=batch_size)
# test_dataset, test_loader = create_dataset_and_loader(X_test, y_test,
#                                                       batch_size=batch_size)
