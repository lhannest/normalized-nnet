from sklearn.datasets import fetch_mldata
import numpy as np

def get_mnist(training_set_size=60000, testing_set_size=10000):
    assert training_set_size + testing_set_size <= 70e3
    mnist = fetch_mldata('MNIST original')
    data = zip(mnist.data, mnist.target)
    np.random.shuffle(data)
    return data[:training_set_size], data[training_set_size:training_set_size + testing_set_size]
