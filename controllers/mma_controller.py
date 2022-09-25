import math
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from .controller import Controller
from models.manipulator_model import ManiuplatorModel


class MMAController(Controller):
    def __init__(self, Tp):
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01
        # III: m3=1.0,  r3=0.3
        self.models = [ManiuplatorModel(Tp, 0.1, 0.05), ManiuplatorModel(Tp, 0.01, 0.01),
                       ManiuplatorModel(Tp, 1., 0.3)]

        self.i = 0
        self.u = np.zeros((2, 2), dtype=np.float32)
        self.x = np.zeros((4, 1))
        self.x_dot = np.zeros((1, 2))
        self.j = 0
        self.error = 100
        self.Tp = Tp

    def choose_model(self, x, u ,x_dot):
        # TODO: Implement procedure of choosing the best fitting model from self.models (by setting self.i)
        prev_error = 1000

        for i, model in enumerate(self.models):
            error = 0.0
            M_inv = np.linalg.inv(model.M(x))
            A = np.concatenate([np.concatenate([[[0.0, 0.0], [0.0, 0.0]], np.eye(2)], 1),
                                np.concatenate([[[0.0, 0.0], [0.0, 0.0]], (-1) * M_inv @ model.C(x)], 1)],
                                0)
            B = np.concatenate([[[0.0, 0.0], [0.0, 0.0]], M_inv], 0)
            x_m = x[:, np.newaxis] + self.Tp * (A @ x[:, np.newaxis] + B @ u)
            x_error = [0.0, 0.0]

            x_error[0] = x_m[2][0]
            x_error[1] = x_m[3][0]
            error = math.sqrt((x_error[0] - x_dot[0])**2) + math.sqrt((x_error[1] - x_dot[1])**2) * 1000.0

            if error < prev_error:
                self.i = i
                prev_error = error

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        # self.choose_model(x)
        q = x[:2]
        q_dot = x[2:]

        if self.j > 0:
            self.choose_model(self.x, self.u, q_dot)

        self.x_dot = q_dot

        K_d = np.array([[1, 0], [0, 1]])
        K_p = np.array([[1, 0], [0, 1]])

        v = q_r_ddot + K_d @ (q_r_dot - q_dot) + K_p @ (q_r - q)    # TODO: add feedback
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v[:, np.newaxis] + C @ q_dot[:, np.newaxis]

        self.u = u
        self.x = x
        self.x_dot = q_dot
        self.j = 1

        return u
