import numpy as np
from cs231n.layers import *
from cs231n.layer_utils import *

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        # Gaussian w/ standard deviation equal to weight_scale.
        self.params["W1"] = np.random.normal(0, weight_scale, (input_dim, hidden_dim))
        self.params["b1"] = np.zeros(hidden_dim,)

        self.params["W2"] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        self.params["b2"] = np.zeros(num_classes,)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        N = X.shape[0]
        scores = None

        # FORWARD PASS
        #########################################

        # Activations and cache for first hidden layer.
        h_1,cache_h1 = affine_forward(X, self.params["W1"], self.params["b1"])
        
        # ReLU for first hidden layer.
        h_1_relu,cache_h1_relu = relu_forward(h_1)

        # Scores and gradient for second hidden layer.
        scores,cache_scores = affine_forward(h_1_relu, self.params["W2"], self.params["b2"])

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        # COMPUTING LOSS AND BACKWARD PASS
        #########################################

        # Softmax cross-entropy loss.
        loss,dscores = softmax_loss(scores, y)

        # Regularisation loss (L2)
        loss += 0.5 * self.reg * (np.sum(self.params["W1"]*self.params["W1"]) 
                + np.sum(self.params["W2"]*self.params["W2"])) # constant 0.5 simplifies derivative

        grads = {}

        # Backprop scores gradient into W2 and b2.
        dh2,grads["W2"],grads["b2"] = affine_backward(dscores, cache_scores)

        # Backprop into ReLU.
        dh1_relu = relu_backward(dh2, cache_h1_relu)

        # Backprop into W1 and B1.
        dh1,grads["W1"],grads["b1"] = affine_backward(dh1_relu, cache_h1)

        # Regularisation gradient contribution.
        grads["W2"] += self.reg * self.params["W2"]
        grads["W1"] += self.reg * self.params["W1"]

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        
        # Initialise parameters for the input layer.
        self.params["W1"] = np.random.normal(0, weight_scale, (input_dim, hidden_dims[0]))
        self.params["b1"] = np.zeros(hidden_dims[0])

        if self.use_batchnorm == True:
            self.params["gamma1"] = np.ones(hidden_dims[0])
            self.params["beta1"] = np.zeros(hidden_dims[0])

        # Initialise parameters for the hidden layers.
        for i in range(2, self.num_layers):
            # Gaussian w/ standard deviation equal to weight_scale.
            self.params["W" + str(i)] = np.random.normal(0, weight_scale, (hidden_dims[i-2], hidden_dims[i-1]))
            self.params["b" + str(i)] = np.zeros(hidden_dims[i-1],)

            if self.use_batchnorm == True:
                self.params["gamma" + str(i)] = np.ones(hidden_dims[i-1])
                self.params["beta" + str(i)] = np.zeros(hidden_dims[i-1])
        
        # Initialise parameters for the scores layer.
        self.params["W" + str(self.num_layers)] = np.random.normal(0, weight_scale, (hidden_dims[self.num_layers-2], num_classes))
        self.params["b" + str(self.num_layers)] = np.zeros(num_classes,)

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test).
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param["mode"] = mode

        N = X.shape[0]
        scores = None

        # FORWARD PASS
        #########################################
        
        activations = X
        h_caches = {}
        h_bn_caches = {}
        h_d_caches = {}

        # Pass through the hidden layers.
        for i in range(1, self.num_layers):
            if not self.use_batchnorm:
                activations,h_caches[i] = affine_relu_forward(activations, self.params["W" + str(i)], self.params["b" + str(i)])
            else:
                activations,h_bn_caches[i] = affine_bn_relu_forward(activations, self.params["W" + str(i)], self.params["b" + str(i)],
                self.params["gamma" + str(i)], self.params["beta" + str(i)], self.bn_params[i-1])

            if self.use_dropout:
                activations,h_d_caches[i] = dropout_forward(activations, self.dropout_param)

        # Pass through the scores layer.
        scores,scores_cache = affine_forward(activations, self.params["W" + str(self.num_layers)], self.params["b" + str(self.num_layers)])
                
        # If test mode return early
        if mode == "test":
            return scores

        # COMPUTING LOSS AND BACKWARD PASS
        #########################################

        # Softmax cross-entropy loss.
        loss,dscores = softmax_loss(scores, y)

        # Regularisation loss (L2)
        loss += 0.5 * self.reg * (np.sum(self.params["W1"]*self.params["W1"]) 
                + np.sum(self.params["W2"]*self.params["W2"])) # constant 0.5 simplifies derivative

        
        dx, grads = dscores, {}

        # Pass through the last layer.
        dx, dW, db = affine_backward(dscores, scores_cache)
        grads["W" + str(self.num_layers)] = dW
        grads["b" + str(self.num_layers)] = db

        # Pass through the hidden layers.
        for i in range(self.num_layers - 1, 0, -1):
            if self.use_dropout:
                dx = dropout_backward(dx, h_d_caches[i])

            if not self.use_batchnorm:
                dx,dW,db = affine_relu_backward(dx, h_caches[i])
            else:
                dx,dW,db,grads["gamma" + str(i)],grads["beta" + str(i)] = affine_bn_relu_backward(dx,h_bn_caches[i])

            # Assign to dictionary with regularisation.
            grads["W" + str(i)] = dW + self.reg * self.params["W" + str(i)]
            grads["b" + str(i)] = db

        return loss, grads

def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    """
    Forward pass for an affine layer with ReLU and batch normalisation. 

    Input:
    - x: Data of shape (N, D)
    - w: Array of weights, shape (D, M)
    - b: Array of biases, shape (M,)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift parameter of scale (D,)
    - bn_param: Dictionary with the following keys:
        - mode: "train" or "test"; required
        - eps: Constant for numeric stability
        - momentum: Constant for running mean / variance.
        - running_mean: Array of shape (D,) giving running mean of features
        - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backwards pass.
    """
    affine_out,fc_cache = affine_forward(x, w, b)
    bn_out,bn_cache = batchnorm_forward(affine_out, gamma, beta, bn_param)
    relu_out,relu_cache = relu_forward(bn_out)
    cache = (fc_cache, bn_cache, relu_cache)

    return relu_out, cache

def affine_bn_relu_backward(dout, cache):
    """
    Backward pass for the above.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
        - x: Input data, of shape (N, d_1, ... d_k)
        - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    fc_cache,bn_cache,relu_cache = cache

    drelu_out = relu_backward(dout, relu_cache)
    dbn_out,dgamma,dbeta = batchnorm_backward(drelu_out, bn_cache)
    dx,dw,db = affine_backward(dbn_out, fc_cache)

    return dx, dw, db, dgamma, dbeta





        
