import os 
import pickle
import numpy as np

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10_data(return_as='array', scaling=True, cifar_path="CIFAR-10"):
    # Initialize variables
    X_train, y_train = [], []
    X_test, y_test = [], []

    # Load all the paths of the pickle files
    
    files_path = os.listdir(cifar_path)

    # Load training data
    for file in files_path: 
        filepath = os.path.join(cifar_path, file)
        if file.startswith("data_batch"):
            temp_dict = unpickle(filepath)
            X_train.extend(temp_dict[b'data'])
            y_train.extend(temp_dict[b'labels'])

    # Load testing data
    for file in files_path:
        filepath = os.path.join(cifar_path, file)
        if file.startswith("test_batch"):
            temp_dict = unpickle(filepath)
            X_test.append(temp_dict[b'data'])
            y_test.extend(temp_dict[b'labels'])

    # Turn as numpy array
    X_train = np.vstack(X_train)
    y_train = np.array(y_train)

    X_test = np.vstack(X_test)
    y_test = np.array(y_test)
    
    # Create X_val set from the training data
    X_train, X_val, y_train, y_val = train_test_split(  X_train, 
                                                        y_train, 
                                                        test_size=0.2,
                                                        random_state=13)
    if scaling:
        # Scale the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    if return_as == 'array':
        pass

    elif return_as == 'tensor': 
        # Reshape into (N, C, H, W) format
        X_train = np.vstack(X_train).reshape(-1, 3, 32, 32).astype(np.float32) #-1 is the number of samples/images, 3 is the channnels, 32 is the height and 32 is the width
        X_val = np.vstack(X_val).reshape(-1, 3, 32, 32).astype(np.float32)
        X_test = np.vstack(X_test).reshape(-1, 3, 32, 32).astype(np.float32)


        # Turn into torch tensor
        X_train, y_train = torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long)
        X_val, y_val = torch.tensor(X_val), torch.tensor(y_val, dtype=torch.long)
        X_test, y_test = torch.tensor(X_test), torch.tensor(y_test, dtype=torch.long)

    else: 
        raise ValueError('return_as should be either array or tensor')
    
    print(f'Data loaded succesfully! as {type(X_train)}')
    print(f'Training data shape: {X_train.shape}')
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_and_create_loaders(cifar_path, return_ds:bool=False):
    # Load the data
    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10_data(return_as='tensor', scaling=True, cifar_path=cifar_path)

    # Create datasets and loaders
    batch_size = 64

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    if return_ds:
        return train_dataset, val_dataset, test_dataset
    else:
        return train_loader, val_loader, test_loader