import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # Compute the loss and the gradient.
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W) 
        correct_class_score = scores[y[i]]

        # Iterate over incorrect classes.
        for j in range(num_classes):
            if j == y[i]:
                continue
            # If margin is met, accumulate loss for the jth example 
            # and calculate associated gradient. 
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin

                # dL/dwj = xi (if margin satisfied); 
                dW[:,j] += X[i,:] 
                
                # dL/dwyi = xi (if margin satisfied); 
                dW[:,y[i]] -= X[i,:] 

    # Convert loss and gradient from sum to average.
    loss /= num_train
    dW /= num_train

    # Regularisation.
    loss += reg * np.sum(W * W)
    dW += reg * W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero
    num_train = X.shape[0]

    # Compute the matrix of all class scores.
    scores = X.dot(W)
    
    # Scores for correct classes; arange gives us the values in scores at the 
    # indices specified in y.
    yi_scores = scores[np.arange(scores.shape[0]),y] 

    # Calculate margins.
    margins = np.maximum(0, scores - np.reshape(yi_scores, (-1,1)) + 1) 
    
    # Set margins for correct classes to 0 (because sum is over incorrect classes).
    margins[np.arange(num_train),y] = 0 
    
    # Calculate average loss across margins.
    loss = np.mean(np.sum(margins, axis=1))

    # Add regularisation.
    loss += reg * np.sum(W*W)

    # Indicator function; 1 for elements (1 class for 1 xi) in which margin is met, otherwise 0.
    binary = margins
    binary[margins > 0] = 1 
    
    # Sum across each column (1 sum per xi).
    row_sum = np.sum(binary, axis=1)

    # Assign negative values to correct classes (yi components) at indices specified by y.
    binary[np.arange(num_train), y] = -row_sum.T

    # Multiply by x.
    dW = np.dot(X.T, binary)

    # Average.
    dW /= num_train

    # Regularise.
    dW += reg*W

    return loss,dW


