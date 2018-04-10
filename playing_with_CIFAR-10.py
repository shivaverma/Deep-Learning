import numpy as np
import cPickle
from numeric_gradient_optimization import gradient_descent


def unpickle(f):
    with open(f, 'rb') as fo:
        d = cPickle.load(fo)
    return d


def unpack():

    x_tr, y_tr, x_te, y_te = None, None, None, None
    for i in range(1, 6):
        dic = unpickle('cifar-10-batches-py/data_batch_'+str(i))
        if x_tr is None:
            x_tr = dic['data']
            y_tr = dic['labels']
        else:
            x_tr = np.concatenate((x_tr, dic['data']), axis=0)
            y_tr = np.concatenate((y_tr, dic['labels']), axis=0)
    dic = unpickle('cifar-10-batches-py/test_batch')
    x_te = dic['data']
    y_te = dic['labels']
    return x_tr, y_tr, x_te, y_te


if __name__ == '__main__':

    x_train, y_train, x_test, y_test = unpack()
    weights = gradient_descent(x_train, y_train, 10)
    print weights

