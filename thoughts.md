# Loading the CIFAR-10 dataset
I trained 27 structures with early stop when for 50 epochs there was no increase found in the validation accuracy. Then I tested the same architectures with one single epoch and checked the same metric. 
    The results were interesting, out of the top 5 architectures found in the 50-epochs-without-increase training, 4 were identified as well as the 5 most accurate in the One-Epoch approach. This supports the method of using such a strategy to reduce computational load since the 50 epochs training took aproximmately 4 hours, against 4 minutes in the One-Epoch.
Perhaps those architectures with the highest accuracies could be used as the basis for next steps in training. 

# Ideas
- Perhaps an early stopping of testing the architectures could be implemented once a certain % of all the pre-defined architectures are tested and none achieve higher results than the best one found yet. **Optimal stopping problem**
- If the top 5 architectures are later on tested with different parameters, isn't that a sort of overly simplified evolutionary algorithm?
- What will happen if each parameter found for the best architecture is isolated? 
- What's the most time expensive parameter? What is its tradeoff? 
- How to define the search space? 
- Is there any property from the training set that could be used to delimit the search space?
- What about the solver and the activation functions?
- What about a progressive approach to the search space? Perhaps start by the fastest training parameters, and out of those isolate them and combine them, get the most accurate one and proceed to change other parameters
- Could it be possible to define the size of each layer based on the top 5 results layer sizes?
- What about concatenating the top architectures?
- What about mixing or alternating the pre-trained layers of the top layers?



# Optimal Stopping Problem
The implementation of the optimal stopping problem can effectively reduce the time of identification of the best architecure by approximately 50% in most scenarios. This, combined with the One Epoch Training approach and GPU support, enables the execution of 27 architectures that previously required 5 hours to be completed in just 15 seconds.


# Learning Curve
The calculation of the Learning Curve via ES required 5 min for a single architecture:

```python
    # Architecture:
    input_size = 32 * 32 * 3  # 3072 features per image
    hidden_layers = [1000, 1000] 
    num_classes = 10  # CIFAR-10 has 10 classes
    lr = 0.001

    # Dataset sizes:
    DATASET_SIZES = 20
    training_sizes = np.logspace(1, np.log10(full_train_size), DATASET_SIZES, dtype=int)
```