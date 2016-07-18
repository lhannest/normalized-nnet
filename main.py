import numpy as np
from mnist import get_mnist
from nnet import build_nnet
import matplotlib.pylab as plt

NORMALIZE = True

hidden_layer_sizes = [50, 50] #np.random.randint(30, 100, np.random.randint(0, 5))
eval_nnet, train_nnet = build_nnet(hidden_layer_sizes, normalize_layers=NORMALIZE)

training_set, testing_set = get_mnist()

avg_error = 0
errors = []
for i, (x, t) in enumerate(training_set):
    print '\rtraining', str(i * 100 / len(training_set)) + '%',
    avg_error += train_nnet(x, t, 1)
    if i%1000 == 0:
        errors.append(avg_error / 1000.)
        avg_error = 0

print

error = 0
for i, (x, t) in enumerate(testing_set):
    print '\rtesting', str(i * 100 / len(testing_set)) + '%',
    prediction, confidence = eval_nnet(x)
    if prediction != t:
        error += 1

print

print 'network topology:', [784] + hidden_layer_sizes + [10]
print 'normalized layer inputs:', NORMALIZE
print 'testing error:', error * 100. / len(testing_set), '%',

plt.plot(errors)
plt.show()
