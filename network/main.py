import network
import cPickle
import gzip
import numpy as np

def load_data_wrapper():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    training_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
    training_results = [vectorized_result(y) for y in training_data[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in validation_data[0]]
    validation_data = zip(validation_inputs, validation_data[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_data = zip(test_inputs, test_data[1])
    return (training_data, validation_data, test_data)


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


training_data, validation_data, test_data = load_data_wrapper()
net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
