# random weights optimization. N images, P pixel, C class.
# we increase weights slightly, if it leads to lesser loss we modify our weights.

# -------------------------------------
#              Img  |   W    | Scores |
# -------------------------------------
#            [N * P].[P * C] = N * D  |
# -------------------------------------


import numpy as np
from hinge_loss import hinge_loss


def random_local_search(x, y, c):

    best_loss = float("inf")
    n, p = np.shape(x)
    best_w = np.random.randn(p, c)*0.01
    step_size = .0001
    for i in xrange(10000):
        w_try = best_w + np.random.randn(p, c)*step_size
        loss = hinge_loss(x, y, w_try)
        if loss < best_loss:
            best_loss = loss
            best_w = w_try
            print best_loss
    return best_w


if __name__ == '__main__':

    x = np.array([[12, 20, 11], [10, 21, 21], [31, 35, 13], [51, 14, 13]])   # 4 images, each of 3 pixel
    y = np.array([0, 1, 1, 2])
    random_local_search(x, y, 3)
