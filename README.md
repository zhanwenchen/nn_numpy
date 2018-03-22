# Neural Network with NumPy

A simple logistic regression library built with NumPy.

## Example Usage

```python
from network import Network

train_X = train_images # (28*28*1=784, 60000)
train_Y = train_labels # (1, 60000)
test_X = test_images # (28*28*1=784, 10000)
test_Y = test_labels # (1, 10000)
num_hidden_layers = 3
layer_width = 5
learning_rate = 0.1
num_iterations = 3000

nn = Network(train_X, train_Y, num_hidden_layers, layer_width)
costs = nn.fit(train_X, train_Y, learning_rate, num_iterations)
print("test accuracy = %s" % (nn.evaluate(test_X, test_Y)))

# Visualize for a single
test_example_5 = test_X[:,4]
predicted_label_for_one_example = nn.predict(test_example_5)
print("For one example, the nn predicted", predicted_label_for_one_example)
```
