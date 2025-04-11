# Steps

I want to create an **efficient, but simple** way for random neurons to be tested, and compared, taking into account that the less amount of training time tends to imply lower final performance.

Up until now, I have tested the OE strategy as the key strategy to achieve such a goal. Nevertheless, my results show that this can be used as a early discarding technique, rather than a top model selection one. 

The threshold for the prune (per epoch) proposed based on the experiments is 15%. 

It appears that my project is leaning towards an overly simplified evolutionary algorithm, Naive if I may.

Previously I've tested for early predictors of the final canonical performance of a Neural Network in the first epochs. Not so many good things there, beyond the threshold but...

My current focus in this moment is to develop the NAS module. That is:
 - A library that builds neural network architectures *-based on a search space?*
 - Allows them to be trained by an arg number of epochs. 
 - Measures their performance, epoch by epoch. 
 - Discards those neurons that correspond to the lowest 15% of all the architectures tested, for an arg number of epochs. 
 - Continues testing the previously selected neurons, up until a condition is met. *-e.g when only 10 candidates remain, pick and return the best model*

# Also:

- Let's test for classification problems versus other models.
- Let's test using the optimal stopping problem solution.

## Previous random thoughts
### Focusing on what's important
So, 
I want to create a way for models to get trained fast and detect fast which one will be a competitive one, even with other models, automatically.

# Random Thoughts

- What about the batch size? For now I'll keep it big because faster.

## Potential Benefits of Decreasing Batch Size Over Epochs:
![image.png](attachment:image.png)

**Hyperparameter Adjustments:** If reducing batch size, consider adjusting the learning rate accordingly (e.g., increase learning rate slightly when batch size decreases).

**Incorporating Model Complexity in Selection**
Instead of only ranking by raw performance, you could penalize architectures with excessive parameters (to favor efficiency).


# Now, how to create the testing for the identified architectures? 
## Extract the best models
Done
## Create the model based on those parameters
Pending
## Train the model by ES
Pending
## Store their results
Pending

# Analyze the difference bw those ones and a random search. 
Hella Pending