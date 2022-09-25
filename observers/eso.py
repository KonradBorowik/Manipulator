from copy import copy
import numpy as np
from models.manipulator_model import ManiuplatorModel


class ESO:
    def __init__(self, A, B, W, L, state, Tp):
        self.model = ManiuplatorModel(Tp, 0.1, 0.05)
        self.A = A
        self.B = B
        self.W = W
        self.L = L
        self.state = np.pad(np.array(state), (0, A.shape[0] - len(state)))
        self.Tp = Tp
        self.states = []
        self.e = [[0.], [0.], [0.]]
        self.tmp = 0

    def set_B(self, i, q):
        b = self.model.M(q)
        b = np.linalg.inv(b)
        b = b[i][i]

        self.B = np.array([0., b, 0.])[:, np.newaxis]


    def update(self, q, u, i, state):
        self.set_B(i, state)
        self.states.append(copy(self.state))
        ### TODO implement ESO update
        estimation = self.get_state()

        e = q[0] - self.W.dot(estimation)

        A = self.A.dot(estimation)
        B = self.B.dot(u)
        C = self.L.dot(e)

        A = A
        B = B.reshape(3, )
        C = C.reshape(3, )

        z_dot = A + B + C

        if self.tmp == 0:
            self.state = z_dot.dot(self.Tp)
            self.tmp = 1
        else:
            self.state = self.state + z_dot.dot(self.Tp)

    def get_state(self):
        return self.state
