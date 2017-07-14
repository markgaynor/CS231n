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
        
        # Initialise parameters for the hidden layers.
        for i in range(1, self.num_layers - 1):
            # Gaussian w/ standard deviation equal to weight_scale.
            self.params["W" + str(i)] = np.random.normal(0, weight_scale, (input_dim, hidden_dims[i-1]))
            self.params["b" + str(i)] = np.zeros(hidden_dims[i-1],)

            if self.use_batchnorm == True:
                self.params["gamma" + str(i)] = np.ones(hidden_dims[i-1])
                self.params["beta" + str(i)] = np.zeros(hidden_dims[i-1])
        
        # Initialise parameters for the scores layer.
        self.params["W" + str(self.num_layers - 1)] = np.random.normal(0, weight_scale, (hidden_dims[self.num_layers-2], num_classes))
        self.params["b" + str(self.num_layers - 1)] = np.zeros(num_classes,)

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
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

        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you"ll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you"ll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        N = X.shape[0]
        scores = None

        # FORWARD PASS
        #########################################

        # Pass through the hidden layers.
        for i in range(1, self.num_layers - 1):
            


            # Gaussian w/ standard deviation equal to weight_scale.
            self.params["W" + str(i)] = np.random.normal(0, weight_scale, (input_dim, hidden_dims[i-1]))
            self.params["b" + str(i)] = np.zeros(hidden_dims[i-1],)

            if self.use_batchnorm == True:
                self.params["gamma" + str(i)] = np.ones(hidden_dims[i-1])
                self.params["beta" + str(i)] = np.zeros(hidden_dims[i-1])
        
        # Initialise parameters for the scores layer.
        self.params["W" + str(self.num_layers - 1)] = np.random.normal(0, weight_scale, (hidden_dims[self.num_layers-2], num_classes))
        self.params["b" + str(self.num_layers - 1)] = np.zeros(num_classes,)
        i = 1

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

        """

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
        """

        return loss, grads

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don"t forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don"t need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
