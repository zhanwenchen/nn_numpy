# network.py
# ----------
# @description A simple fully connected (dense) neural network using NumPy
# @author Zhanwen "Phil" Chen {zhanwen.chen}@vanderbilt.edu
# @date March 22, 2018
# @version 0.1.0

# Notations
# ---------
# For a single neuron in the first layer,
# Wx + b = Z (W should strictly be W[0,0] and b b[0,0]);
# A = relu(Z);

# In later layers there's no more X but A of the previous layer, so
# WA[-1] + b = Z
# A = relu(Z);

__all__ = ['Network']

import numpy as np

np.seterr(all="raise")

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
        if cost_function == "cross_entropy":
            self.cost = cross_entropy

        # 5. Initialize weights and biases with randomization method.
        if random_method == "normal":
            self.W = [np.random.randn(num_out, num_in) * weight_scale \
                        for num_in, num_out in self.layers]
            # BUG: THIS IS DIFFERENT FROM db in backprop
            self.b = [np.zeros((num_out, 1)) for num_in, num_out in self.layers]


    def fit(self, train_X, train_Y, learning_rate, num_iterations,
            activation = "relu", cost = "cross-entropy"):
        """
        Main method for running the gradient descent algo.

        Args:
            train_X: training features. IMPORTANT: X must be formatted as a column vector
                such that each column is one example and one row is one feature.

            train_Y: training labels. IMPORTANT: Y must be formatted as a column vector
                such that each column is one example and one row is one label.

            learning_rate: The learning rate for gradient descent.
            num_iterations: The number of times we go through all training examples.

        Returns:
            costs: A Python array of training costs.
        """

        # 1. Forward Propagation with initial weights
        Z, A = self.forward_prop(train_X)

        init_cost = self.cost(A[-1], train_Y)
        costs = [init_cost]

        # 2. Run learning
        for i in range(num_iterations):

            #1. Backprop
            dW, db = self.backprop(train_X, train_Y, Z, A)

            #3. Update weights and biases for each layer
            self.W = [W_l - learning_rate * dW_l for W_l, dW_l in zip(self.W, dW)]
            self.b = [b_l - learning_rate * db_l for b_l, db_l in zip(self.b, db)]

            #4. Forward Prop with updated weights
            Z, A = self.forward_prop(train_X)

            current_cost = self.cost(A[-1], train_Y)
            print("%sth iter: cost = %s" % (i, current_cost))
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

    def predict(self, X):
        """
        Method for predicting the labels given new features, after training.

        Args:
            X: prediction features. IMPORTANT: X must be formatted as a column vector
                such that each column is one example and one row is one feature.

        Returns:
            predicted_labels: The predicted labels given new features.
        """
        Z, A = self.forward_prop(X)
        Y_hat = A[-1]
        predicted_labels = np.rint(Y_hat)
        return predicted_labels

    def forward_prop(self, X):
        """
        Populate network with hypothesis and activations.

        Args:
            X: Training, validation, or testing features in column-vector form
                of shape (num_features, num_examples).

        Returns:
            Z: A (Python) list of hypothesis by layer.
            A: A (Python) list of activations by layer, corresponding to Z.

        Raises:
            ValueError: Raised if train_X, train_Y have incorrect shapes.
        """
        # 1. Initialize hypotheses and activations
        Z = [np.zeros((len(w[0]), self.m)) for w in self.W]
        A = [np.zeros((len(w[0]), self.m)) for w in self.W]

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


    def backprop(self, train_X, train_Y, Z, A):
        """
        Auxillary method for back propagation.

        Args:
            train_X: training features. IMPORTANT: X must be formatted as a column vector
                such that each column is one example and one row is one feature.

            train_Y: training labels. IMPORTANT: Y must be formatted as a column vector
                such that each column is one example and one row is one label.

            Z: A (Python) list of hypothesis by layer, from forward_prop.
            A: A (Python) list of activations by layer, from forward_prop.

        Returns:
            dW: A (Python) list of gradients corresponding to self.W exactly.
            db: A (Python) list of gradients corresponding to self.b exactly.
        """
        # 1. Initialize arrays dZ, dW, db
        dZ = [np.zeros((len(w[0]), self.m)) for w in self.W]
        dW = [np.zeros((num_out, num_in)) for num_in, num_out in self.layers]
        db = [np.zeros((num_out, 1)) for _, num_out in self.layers]

        if len(self.layers) == 1:
            dZ[0] = A[-1] - train_Y
            dW[0] = np.dot(dZ[0], train_X.T)/self.m
            db[0] = np.expand_dims(np.sum(dZ[0], axis=1)/self.m, axis=1)
        else:
            # 2. Calculate the gradients for the last (output) layer
            dZ[-1] = A[-1] - train_Y # NOTE: True iff last layer has sigmoid activation.
            dW[-1] = np.dot(dZ[-1], A[-1-1].T)/self.m  # (1, 2)
            db[-1] = np.expand_dims(np.sum(dZ[-1], axis=1)/self.m, axis=1) # REVIEW or np.average()

            # 3. Calculate the gradients for the hidden layers backwards.
            for i in range(-2, -len(self.layers), -1):
                dZ[i] = np.dot(self.W[i+1].T, dZ[i+1]) * self.sigma_derivative(Z[i])
                db[i] = np.expand_dims(np.sum(dZ[i], axis=1)/self.m, axis=1)
                dW[i] = np.dot(dZ[i], A[-i-1].T)/self.m

            # 4. Calculate the gradients for the first (input) layer
            dZ[0] = np.dot(self.W[1].T, dZ[1]) * self.sigma_derivative(Z[0])
            db[0] = np.expand_dims(np.sum(dZ[0], axis=1)/self.m, axis=1)
            dW[0] = np.dot(dZ[0], train_X.T)/self.m

        return dW, db





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
    loss = -(Y * np.log(Y_hat) + (1.0 - Y) * np.log(1.0 - Y_hat))
    cost = np.average(loss)
    return cost

def relu(Z):
    return Z * (Z > 0)

def relu_derivative(Z):
    return (Z > 0) * 1.0

def sigmoid(Z):
    return 1.0 / (1.0 + np.exp(-Z))
