from copy import copy
import numpy as np


class ESO:
    def __init__(self, A, B, W, L, state, Tp):
        self.A = A
        self.B = B
        self.W = W
        self.L = L
        self.state = np.array([0., 0., 0., 0., 0., 0.])
        self.Tp = Tp
        self.states = []
        self.e = [[0.], [0.], [0.]]
        self.tmp = 0

    def set_B(self, B):
        self.B = B

    def update(self, q, u):
        self.states.append(copy(self.state))
        ### TODO implement ESO update
        estimation = self.get_state()

        e = q[0:2].transpose() - self.W.dot(estimation)
        A = self.A.dot(estimation)
        B = self.B.dot(u)
        C = self.L.dot(e)

        z_dot = A + B + C

        if self.tmp == 0:
            self.state = z_dot.dot(self.Tp)
            self.tmp = 1
        else:
            self.state = self.state + z_dot.dot(self.Tp)

    def get_state(self):
        return self.state
