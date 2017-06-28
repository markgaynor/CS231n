import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """

    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]

    # For each training example:
    for i in range(num_train):
        # Calculate scores.
        scores = X[i].dot(W)
            
        # Normalisation trick for stability: shift 
        # the values so that the highest number is 0.
        scores -= np.max(scores)

        # Accumulate loss for that training example (Li = -fyi + log sigma efi).
        loss += -scores[y[i]] + np.log(sum(np.exp(scores)))
        
        # For each class
        for j in range(W.shape[1]):
            # Softmax output for that class.
            output = np.exp(scores[j]) / sum(np.exp(scores))
            # Incorrect class:
            if j != y[i]:
                # Accumulate gradient.
                dW[:,j] += output * X[i]
            # Correct class.
            else:
                # Accumulate gradient.
                dW[:,j] += (output - 1) * X[i]
    
    # Convert from sum to average.
    loss /= num_train
    dW /= num_train

    # Regularisation.
    loss += reg * np.sum(W * W)
    dW += reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]

    # Compute the matrix of class scores.
    scores = X.dot(W)

    # Normalisation trick for stability.
    scores -= np.max(scores, axis=1).reshape(-1,1)
    
    # Li = -fyi + log sigma efj
    fyi = scores[np.arange(num_train), y] # Scores for correct classes.
    sigma_efj = np.sum(np.exp(scores), axis=1)
    loss = np.sum(-fyi + np.log(sigma_efj)) # Sum to compute total loss for each example.
    
    # Softmax outputs for each class.
    output = np.exp(scores) / np.reshape(sigma_efj, (-1,1))

    # Subtract 1 from outputs for correct classes (equivalently to naive computation above).
    output[np.arange(num_train),y] -= 1
    
    # Calculate gradient.
    dW = X.T.dot(output)

    # Convert from sum to average.
    loss /= num_train
    dW /= num_train

    # Regularisation.
    loss += reg * np.sum(W * W)
    dW += reg * W

    return loss, dW
