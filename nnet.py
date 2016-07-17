import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import theano.tensor.extra_ops as extra
import numpy as np

def layer(X, input_size, output_size, activation=nnet.softplus):
    r = np.sqrt(6. / (input_size + output_size))
    shape = (input_size + 1, output_size)
    w = np.asarray(np.random.uniform(-r, r, shape), dtype=theano.config.floatX)
    W = theano.shared(w)
    X = T.concatenate([X, [1]])
    Z = T.dot(X, W)
    return activation(Z), W

def normalize(X):
    A = X / T.max(X)
    return A - T.mean(A)

def build_nnet(layer_sizes, normalize_layers=False):
    X = T.vector(dtype='float32')
    t = T.scalar(dtype='int32')
    alpha = T.scalar(dtype='float32')
    t_onehot = extra.to_one_hot(t.reshape((1, 1)), 10)

    weights = []

    # We always want to normalize the inputs to the first layer
    Y, W = layer(normalize(X), 784, layer_sizes[0])
    weights.append(W)

    for l1, l2 in zip(layer_sizes[1:-1], layer_sizes[2:]):
        if normalize_layers:
            Y = normalize(Y)
        Y, W = layer(Y, l1, l2)
        weights.append(W)

    if normalize_layers:
        Y = normalize(Y)
    Y, W = layer(Y, layer_sizes[-1], 10, activation=nnet.softmax)
    weights.append(W)

    mse = T.mean(T.sqr(Y - t_onehot))
    updates = [(W, W - alpha * T.grad(cost=mse, wrt=W)) for W in weights]

    prediction = T.argmax(Y)
    confidence = T.max(Y)

    eval_nnet = theano.function(inputs=[X], outputs=[prediction, confidence])
    train_nnet = theano.function(inputs=[X, t, alpha], outputs=mse, updates=updates)

    return eval_nnet, train_nnet
