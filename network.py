# network.py
# ----------
# @description A simple fully connected (dense) neural network using NumPy
# @author Zhanwen "Phil" Chen {zhanwen.chen}@vanderbilt.edu
# @date March 22, 2018
# @version 0.1.0

# NOTE: The cost function is cross entropy. Output layer is hard-coded to use
# sigmoid activation, making its gradient (A-y) iff the cost function is
# cross entropy.

# Notations
# ---------
# For a single neuron in the first layer,
# Wx + b = Z (W should strictly be W[0,0] and b b[0,0]);
# A = relu(Z);

# In later layers there's no more X but A of the previous layer, so
# WA[-1] + b = Z
# A = relu(Z);


# BUG: Alternatively, instead of implementing dropout in forward_prop,
#      make it a separate function in the fit function, so that we can reuse
#      forward_prop results.

# BUG: For dropout, should I accumulate DURING forward_prop (e.g. Z[1] = A[0] * D[0] ...)
#      or AFTER forward_prop (e.g., A = [A_l * D_l for A_l, D_l in zip(A, D)])?
__all__ = ['Network']

import numpy as np

# np.seterr(all="raise")

class Network:
    def __init__(self, train_X, train_Y, num_hidden_layers, layer_width,
                activation_method = "relu", cost_function = "cross_entropy",
                random_method = "normal", weight_scale = 0.01):
        """
        Initialize the network with training data and hyperparameters.

        Args:
            train_X: Training features in column-vector form of shape
                (num_features, num_examples).
            train_Y: Training labels in column-vector form of shape
                (num_classes, num_examples).
            num_layers: The number of hidden layers.
            layer_width: The number of neurons for all layers.
            activation_method = "sigmoid": The sigma function for hidden layers.
            random_method = "normal": The randomization method for init W, b.
            weight_scale = 0.01: The standard deviation for randomization.

        Returns:
            Constructed object.

        Raises:
            ValueError: Raised if train_X, train_Y have incorrect shapes.
        """

        # 1. Enforce shapes
        n, m_X = train_X.shape

        num_classes, m_Y = train_Y.shape
        if num_classes >= m_Y:
            raise ValueError("Constructor: Y must be a column vector and have more data points than num_classes, but Y.shape =", train_Y.shape)

        if m_X != m_Y:
            raise ValueError("Constructor: X, Y must have the same number of examples, but X has %s while Y has %s" % (m_X, m_Y))

        self.n = n
        self.m = m_X
        self.num_classes = num_classes

        # 2. Get layer dimensions
        if num_hidden_layers == 0:
            temp_layers = [n, num_classes]
        else:
            hidden_layer_tuple = (layer_width,) * num_hidden_layers
            temp_layers = [n, *hidden_layer_tuple, num_classes]
        self.layers = list(zip(temp_layers[:-1], temp_layers[1:]))

        # 3. Set activation method
        if activation_method == "relu":
            self.sigma = relu
            self.sigma_derivative = relu_derivative
        else:
            raise ValueError("Only ReLU activation has been implemented.")

        # 4. Set cost function
        if cost_function in ["cross_entropy"]:
            self.cost_function = cost_function

        # 5. Initialize weights and biases with randomization method.
        if random_method == "normal":
            self.W = [np.random.randn(num_out, num_in) * weight_scale \
                        for num_in, num_out in self.layers]
            # BUG: THIS IS DIFFERENT FROM db in backprop
            self.b = [np.zeros((num_out, 1)) for num_in, num_out in self.layers]


    def fit(self, train_X, train_Y, learning_rate, num_iterations,
            regularization = None, l=0, activation = "relu",
            cost = "cross-entropy", printing = False,
            dropout=False, keep_prob=1.0):
        """
        Main method for running the gradient descent algo.

        Args:
            train_X: training features. IMPORTANT: X must be formatted as a column vector
                such that each column is one example and one row is one feature.

            train_Y: training labels. IMPORTANT: Y must be formatted as a column vector
                such that each column is one example and one row is one label.

            learning_rate: The learning rate for gradient descent.
            num_iterations: The number of times we go through all training examples.
            l: Lambda - the hyperparameter for L2 Regularization.
            activation = "relu": Use ReLU for *hidden* layer activations.
            cost = "cross-entropy": Use cross_entropy for loss/cost
            printing = False: Whether to print training results each iteration.

        Returns:
            costs: A Python array of training costs.
        """

        if regularization is None and l != 0:
            raise ValueError("Network.fit: You cannot specify lambda without specifying regularization method!")

        self.l = l
        if regularization is None or regularization in ["L2"]:
            self.regularization = regularization

        # 1. Forward Propagation with initial weights
        Z, A = self.forward_prop(train_X)

        if dropout:
            D = [np.random.binomial([np.ones(A_l.shape)], keep_prob)[0] for A_l in A] # same shape as A_l
            A_before_dropout = np.copy(A)
            A = [A_l * D_l / keep_prob for A_l, D_l in zip(A, D)]
            init_cost = self.cost(A_before_dropout[-1], train_Y)
        else:
            # Do not use dropout activations in cost calculation due to zeros.
            init_cost = self.cost(A[-1], train_Y)

        costs = [init_cost]

        # 2. Run learning
        for i in range(num_iterations):

            #1. Backprop
            dW, db = self.backprop(train_X, train_Y, Z, A, l)

            #3. Update weights and biases for each layer
            self.W = [W_l - learning_rate * dW_l for W_l, dW_l in zip(self.W, dW)]
            self.b = [b_l - learning_rate * db_l for b_l, db_l in zip(self.b, db)]

            #4. Forward Prop with updated weights
            Z, A = self.forward_prop(train_X)

            # if dropout:
            #     A_masked, Mask = self.apply_mask(keep_prob)
            # NOTE: The problem was probably with cost, because when we are
            # evaluating, we are using a bunch of zeros in the cost, where
            # we should be using the real one. So evaluating cost needs to
            # to use the Z, A

            # Do not use dropout activations in cost calculation due to zeros.


            if dropout:
                D = [np.random.binomial([np.ones(A_l.shape)], keep_prob)[0] for A_l in A] # same shape as A_l
                A_before_dropout = np.copy(A)
                A = [A_l * D_l / keep_prob for A_l, D_l in zip(A, D)]
                current_cost = self.cost(A_before_dropout[-1], train_Y)
            else:
                # Do not use dropout activations in cost calculation due to zeros.
                current_cost = self.cost(A[-1], train_Y)

            if printing: print("%sth iter: cost = %s" % (i, current_cost))
            costs.append(current_cost)

        return costs

    def evaluate(self, X, Y):
        """
        Method for evaluating the accuracy given features, labels pairs.

        Args:
            X: Evaluation features.
            Y: Evaluation labels.
        Returns:
            accuracy: The percentage of correct predicted labels vs real labels.
        """
        preds = self.predict(X)
        test_results = preds == Y
        accuracy = np.average(test_results)
        return accuracy

    def predict(self, X, activations_only=False):
        """
        Method for predicting the labels given new features, after training.

        Args:
            X: prediction features. IMPORTANT: X must be formatted as a column vector
                such that each column is one example and one row is one feature.

        Returns:
            predicted_labels: The predicted labels given new features.
        """

        # If there's one example, its shape can be (n,) instead of (n,1).
        if X.ndim == 1:
            X = np.expand_dims(X, axis=1)

        Z, A = self.forward_prop(X)
        Y_hat = A[-1]
        if activations_only:
            return Y_hat
        predicted_labels = np.rint(Y_hat)
        return predicted_labels

    def forward_prop(self, X):
        """
        Populate network with hypothesis and activations.

        Args:
            X: Training, validation, or testing features in column-vector form
                of shape (num_features, num_examples).
        Returns:
            Z: A (Python) list of hypotheses by layer.
            A: A (Python) list of activations by layer, corresponding to Z.

        Raises:
            ValueError: Raised if train_X, train_Y have incorrect shapes.
        """
        # 1. Initialize hypotheses and activations
        Z = [np.zeros((len(W_l), self.m)) for W_l in self.W]
        A = [np.zeros((len(W_l), self.m)) for W_l in self.W]

        # 2. Populate all layers
        if len(self.layers) == 1:
            # 2e. Edge Case: if single layer (no hidden layer), then in->out
            Z[0] = np.dot(self.W[0], X) + self.b[0]
            A[0] = sigmoid(Z[0])

        else:
            # 2a. Populat input layer
            Z[0] = np.dot(self.W[0], X) + self.b[0]
            A[0] = self.sigma(Z[0])

            # 2b. Populate all hidden layers
            for i in range(1, len(self.layers) - 1):
                Z[i] = np.dot(self.W[i], A[i-1]) + self.b[i]
                A[i] = self.sigma(Z[i])

            # 2c. Populate last (output) layer with sigmoid (hard-coded) activation.
            Z[-1] = np.dot(self.W[-1], A[-2]) + self.b[-1]
            A[-1] = sigmoid(Z[-1])

        return Z, A

    def backprop(self, train_X, train_Y, Z, A, l, D=None):
        """
        Auxillary method for back propagation.

        Args:
            train_X: training features. IMPORTANT: X must be formatted as a column vector
                such that each column is one example and one row is one feature.

            train_Y: training labels. IMPORTANT: Y must be formatted as a column vector
                such that each column is one example and one row is one label.

            Z: A (Python) list of hypothesis by layer, from forward_prop.
            A: A (Python) list of activations by layer, from forward_prop.
            l: Lambda - the hyperparameter for L2 regularization.
            D=None: if not None, use D as the mask.

        Returns:
            dW: A (Python) list of gradients corresponding to self.W exactly.
            db: A (Python) list of gradients corresponding to self.b exactly.
        """
        # 1. Initialize arrays dZ, dW, db
        dZ = [np.zeros((len(w[0]), self.m)) for w in self.W]
        dW = [np.zeros((num_out, num_in)) for num_in, num_out in self.layers]
        db = [np.zeros((num_out, 1)) for _, num_out in self.layers]

        def dA_sigmoid(A_l):
            return -train_Y/A_l + (1.0-train_Y)/(1.0-A_l)

        # TODO: REVIEW for sigmoid, dA *= D[i] / keep_prob might be unnecessary?
        if len(self.layers) == 1:
            # dA = dA_sigmoid(A[0])
            # if D: dA *= D[0] / keep_prob
            # NOTE: for sigmoid, dA *= D[i] / keep_prob might be unnecessary
            dZ[0] = A[0] - train_Y
            # dZ[0] = dA * sigmoid_derivative(Z[0])
            # dW[0] = np.dot(dZ[0], train_X.T)/self.m + self.l * self.W[0] / self.m # REVIEW: W_l divided by m or not?
            dW[0] = np.dot(dZ[0], train_X.T)/self.m + self.l * self.W[0]
            db[0] = np.sum(dZ[0], axis=1, keepdims=True) / self.m # REVIEW or np.average(
        else:
            # 2. Calculate the gradientor the last (output) layer
            # dA = dA_sigmoid(A[-1])
            # NOTE: for sigmoid, dA *= Di] / keep_prob might be unnecessary
            # if D: dA *= D[-1] / keep_prob
            # dZ[-1] = dA * self.sigma_derivative(Z[-1])
            dZ[-1] = A[-1] - train_Y # NOTE: True iff last layer has sigmoid activation.
            # dW[-1] = np.dot(dZ[-1], A[-1-1].T)/self.m + self.l * self.W[-1] / self.m
            dW[-1] = np.dot(dZ[-1], A[-1-1].T)/self.m + self.l * self.W[-1]
            db[-1] = np.sum(dZ[-1], axis=1, keepdims=True) / self.m

            # 3. Calculate the gradients for the hidden layers backwards.
            for i in range(-2, -len(self.layers), -1):
                dA = np.dot(self.W[i+1].T, dZ[i+1])
                # NOTE: for non-sigmoid, dA *= D[i] / keep_prob is necessary
                if D: dA *= D[i] / keep_prob
                dZ[i] = dA * self.sigma_derivative(Z[i])
                # dW[i] = np.dot(dZ[i], A[-i-1].T)/self.m + self.l * self.W[i] / self.m
                dW[i] = np.dot(dZ[i], A[-i-1].T)/self.m + self.l * self.W[i]
                db[i] = np.sum(dZ[i], axis=1, keepdims=True) / self.m

            # 4. Calculate the gradients for the first (input) layer
            # NOTE: for non-sigmoid, dA *= D[i] / keep_prob is necessary
            dA = np.dot(self.W[1].T, dZ[1])
            if D: dA *= D[i] / keep_prob

            dZ[0] = dA * self.sigma_derivative(Z[0])
            dW[0] = np.dot(dZ[0], train_X.T)/self.m + self.W[0] * self.l / self.m
            dW[0] = np.dot(dZ[0], train_X.T)/self.m + self.W[0] * self.l
            db[0] = np.sum(dZ[0], axis=1, keepdims=True) / self.m


        # [print("avg(dW[%s]) =" % i, dW_l) for i, dW_l in enumerate(dW)]
        return dW, db

    def cost(self, Y_hat, Y):
        if self.cost_function == "cross_entropy":
            cost = cross_entropy(Y_hat, Y)
        if self.regularization in ["L2"]:
            cost += self.penalty(self.regularization)
        return cost


    def penalty(self, which_norm):
        """
        Calculate the sum of norms for all layers using numpy.linalg.norm.
        """

        if which_norm == "L2":
            l2_norm = sum([np.linalg.norm(W_l) for W_l in self.W])
            penalty = self.l * l2_norm / 2.0
        else:
            raise ValueError("norm: only L2 norm is implemented right now. You asked for", which_norm)

        return penalty


# Utilities
def cross_entropy(Y_hat, Y):
    """
    Initialize the network with training data and hyperparameters.

    Args:
        Y_hat: Last (output) layer activations.
        Y: Training, validation, or testing label in column-vector form
            of shape (num_classes, num_examples).

    Returns:
        cost: A float - the cross entropy cost for output layer activations.
    """
    # print("Y_hat =", Y_hat)
    # print("Y_hat.shape =", Y_hat.shape)
    # loss = -(Y * np.log(Y_hat) + (1.0 - Y) * np.log(1.0 - Y_hat))
    loss1 = Y * np.log(Y_hat)
    loss2 = (1.0 - Y) * np.log(1.0 - Y_hat)
    loss = -(loss1 + loss2)
    cost = np.average(loss)
    return cost

def relu(Z):
    return Z * (Z > 0)

def relu_derivative(Z):
    return (Z > 0) * 1.0

def sigmoid(Z):
    return 1.0 / (1.0 + np.exp(-Z))

def sigmoid_derivative(Z):
    sigmoid_Z = sigmoid(Z)
    return sigmoid_Z * (1.0 - sigmoid_Z)
