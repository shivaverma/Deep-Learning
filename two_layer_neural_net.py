# implementation of a sigmoid two layer neural network.
# we train this network providing data and label.

import numpy as np


def train(x, y):

    syn0 = np.random.random((3, 4))        # creating random weights
    syn1 = np.random.random((4, 1))

    for i in xrange(10000):

        l1 = 1/(1+np.exp(-(np.dot(x, syn0))))          # sigmoid activation function
        l2 = 1/(1+np.exp(-(np.dot(l1, syn1))))
        l2_delta = (y-l2)*(l2*(1-l2))                  # derivative of loss
        l1_delta = l2_delta.dot(syn1.T)*(l1*(1-l1))
        syn1 += l1.T.dot(l2_delta)                     # updating weights
        syn0 += x.T.dot(l1_delta)
        print(syn1)


if __name__ == '__main__':

    x = np.array([[0, 0, 1],              # 4 images, each of 3 pixel
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])

    y = np.array([[0, 1, 1, 0]]).T        # transpose of label matrix
    train(x, y)
