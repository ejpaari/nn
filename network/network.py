import random
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


class Network(object):    
    def __init__(self, sizes):
        self.num_layers = len(sizes) # number of neurons in each layer
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]


    """Stochastic gradient descent"""
    def SGD(self, training_data, iterations, mini_batch_size, learning_rate, test_data = None):
        if test_data: 
            n_test = len(test_data)
        n = len(training_data)
        for j in xrange(iterations):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                nb = [np.zeros(b.shape) for b in self.biases]
                nw = [np.zeros(w.shape) for w in self.weights]
                for x, y in mini_batch:
                    delta_nb, delta_nw = self.backprop(x, y)
                    nb = [t_nb + t_delta_nb for t_nb, t_delta_nb in zip(nb, delta_nb)]
                    nw = [t_nw + t_delta_nw for t_nw, t_delta_nw in zip(nw, delta_nw)]
                self.weights = [w - (learning_rate / len(mini_batch)) * nw
                                for w, nw in zip(self.weights, nw)]
                self.biases = [b - (learning_rate / len(mini_batch)) * nb 
                               for b, nb in zip(self.biases, nb)]
            if test_data:
                print("Iteration {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Iteration {0} complete".format(j))


    """Return nb and nw representing the gradient for the cost function"""
    def backprop(self, x, y):
        nb = [np.zeros(b.shape) for b in self.biases]
        nw = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b # feedforward
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = (activations[-1] - y) * sigmoid_derivative(zs[-1])
        nb[-1] = delta
        nw[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_derivative(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nb[-l] = delta
            nw[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nb, nw)


    def feedforward(self, a):        
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a


    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
