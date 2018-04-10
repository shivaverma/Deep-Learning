# this is a small example of backpropogation
# there are 2 inputs x and y passing through a multiply gate.
# output from multiply gate is z.


#   x-------|
#           |        --------
#           ---------| mult |------------ z
#           ---------| gate |
#           |        --------
#   y-------|


class MultiplyGates(object):

    def __init__(self):

        self.x = 0
        self.y = 0

    def forward(self, x, y):

        """ here we are calculating dz/dx and dz/dy """

        z = x * y
        self.x = x      # this is [dz/dy]
        self.y = y      # this is [dz/dx]
        return z        # we can send this z for input of next gate

    def backward(self, dz):

        """ Here dz means dL/dz. We are calculating dx and dy means dL/dx and dL/dy """

        dx = self.y*dz      # dL/dx = dz/dx * dL/dz
        dy = self.x*dz      # dL/dy = dz/dy * dL/dz
        return [dx, dy]     # dx,dy goes to next gate for calculating further gradient
