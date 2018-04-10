# implementation of hinge loss / multiclass svm loss
# without regularization loss

# -------------------------------------
#              Img  |   W    | Scores |
# -------------------------------------
#            [N * P].[P * C] = N * D  |
# -------------------------------------


#          regularization loss
#       W---------------------->
#               |              |
#             Score -------> Loss
#               |              |
#     (x,y)-------------------->


import numpy as np


def hinge_loss(x, y, w):

    delta = 1
    scores = x.dot(w)
    loss, ind = 0., 0
    for i in scores:
        margin = np.maximum(0, i-i[y[ind]]+delta)
        margin[y[ind]] = 0
        loss_i = np.sum(margin)
        loss += loss_i
        ind += 1
    loss /= ind
    return loss


if __name__ == '__main__':

    x = np.array([[3, 1, 6, 3], [1, 3, 2, 1], [3, 2, 4, 1]])             # three images
    w = np.array([[0, 0, 4], [2, 1, 2], [6, 5, 3], [3, 0, 0]])           # weights.  3 * 4. (3 total classes)
    y = np.array([1, 0, 2])                                              # correct label of class (1 is cat)
    print hinge_loss(x, y, w)
