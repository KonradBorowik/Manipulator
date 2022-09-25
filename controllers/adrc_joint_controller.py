import numpy as np
from observers.eso import ESO
from .controller import Controller
from models.manipulator_model import ManiuplatorModel

class ADRCJointController(Controller):
    def __init__(self, b, K_p, K_d, p, q0, Tp):
        self.model = ManiuplatorModel(Tp, 0.1, 0.05)
        self.b = b
        self.K_p = K_p
        self.K_d = K_d
        self.q0 = q0

        l1 = 5 * p
        l2 = 50 * p
        l3 = 500 * p

        A = np.array([[0., 1., 0.], [0., 0., 1.], [0., 0., 0.]])
        B = np.array([0., b, 0.])[:, np.newaxis]
        L = np.array([l1, l2, l3])[:, np.newaxis]
        W = np.array([1., 0., 0.])

        self.eso = ESO(A, B, W, L, q0, Tp)

    def set_b(self, b):
        ### TODO update self.b and B in ESO
        self.b = b

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot, i, state):
        ### TODO implement ADRC
        estimation = self.eso.get_state()
        q_estimated = estimation[0]
        q_d_estimated = estimation[1]

        f = estimation[2]
        v = q_d_dot + self.K_d * (q_d_dot - q_d_estimated) + self.K_p * (q_d - x[0])
        u = (v - f) / self.b

        self.eso.update(x, u, i, state)

        return float(u)