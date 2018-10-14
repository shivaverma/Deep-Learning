# random weights optimization. N images, P pixel, C class.
# here we calculate gradient of loss w.r.t. w(input), and increase
# the weights to the opposite direction of gradient.

# -------------------------------------
#              Img  |   W    | Scores |
# -------------------------------------
#            [N * P].[P * C] = N * D  |
# -------------------------------------

# following formula is used to calculate gradient:

# -----------------------------------------------
#        df(x)                 f(x+h) - f(x)    |
#       -------  =  Lim(h->0) ---------------   |
#         dx                          h         |
# -----------------------------------------------


import numpy as np
from hinge_loss import hinge_loss


def evaluate_gradient(x, y, w):

    fw = hinge_loss(x, y, w)
    grad = np.zeros(w.shape)
    h = 0.0001
    it = np.nditer(w, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ind = it.multi_index
        old_value = w[ind]
        w[ind] += h
        fwh = hinge_loss(x, y, w)
        w[ind] = old_value
        grad[ind] = (fwh-fw)/h
        it.iternext()
    return grad


def gradient_descent(x, y, c):

    n, p = np.shape(x)
    w = np.random.randn(p, c)*0.001
    step_size = .001
    print('original loss: %f' % hinge_loss(x, y, w))
    for i in xrange(10):
        df = evaluate_gradient(x, y, w)
        w = w - df*step_size
        loss = hinge_loss(x, y, w)
        print(loss)
    return w


if __name__ == '__main__':

    x = np.array([[12, 20, 11], [10, 21, 21], [31, 35, 13], [51, 14, 13]])   # 4 images, each of 3 pixel
    y = np.array([0, 1, 1, 2])
    gradient_descent(x, y, 3)
