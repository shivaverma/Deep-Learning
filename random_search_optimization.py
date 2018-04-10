# random weights optimization. N images, P pixel, C class.
# we randomly search the weights, and select one which gives lowest loss

# -------------------------------------
#              Img  |   W    | Scores |
# -------------------------------------
#            [N * P].[P * C] = N * D  |
# -------------------------------------


import numpy as np
from hinge_loss import hinge_loss


def random_search(x, y, c):

    best_loss = float("inf")
    n, p = np.shape(x)
    best_w = np.random.randn(p, c)*0.01
    for i in xrange(10):
        w = np.random.randn(p, c)*0.01
        loss = hinge_loss(x, y, w)
        if loss < best_loss:
            best_loss = loss
            best_w = w
            # print best_loss
    return best_w


if __name__ == '__main__':

    x = np.array([[12, 20, 11], [10, 21, 21], [31, 35, 13], [51, 14, 13]])   # 4 images, each of 3 pixel
    y = np.array([0, 1, 1, 2])
    random_search(x, y, 3)
