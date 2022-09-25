import numpy as np

# from models.free_model import FreeModel
from observers.eso_flc import ESO
from .adrc_joint_controller import ADRCJointController
from .controller import Controller
# from models.ideal_model import IdealModel
from models.manipulator_model import ManiuplatorModel


class ADRFLController(Controller):
    def __init__(self, Tp, q0, K_p, K_d, p):
        self.model = ManiuplatorModel(Tp, 0.1, 0.05)
        self.K_p = K_p
        self.K_d = K_d
        A = np.array([[0., 1., 0.], [0., 0., 1.], [0., 0., 0.]])
        B = np.array([0., 10., 0.])[:, np.newaxis]
        W = np.array([[1., 0., 0., 0., 0., 0.], [0., 1., 0., 0., 0., 0.]])

        l1 = 5 * p
        l2 = 50 * p
        l3 = 500 * p
        self.L = np.array([[l1[0], 0.], [0., l1[1]], [l2[0], 0.], [0., l2[1]], [l3[0], 0.], [0., l3[1]]])

        self.eso = ESO(A, B, W, self.L, q0, Tp)
        self.update_params(q0[:2], q0[2:])

    def update_params(self, q, q_dot):
        ### TODO Implement procedure to set eso.A and eso.B
        state = [q[0], q[1], q_dot[0], q_dot[1]]

        mtmp = self.model.M(state)
        ctmp = self.model.C(state)

        b = np.linalg.inv(mtmp)
        if np.all(ctmp):
            a = np.linalg.inv(ctmp)
            atmp = -b.dot(a)

            self.eso.A = np.array([[0., 0., 1., 0., 0., 0.],
                                   [0., 0., 0., 1., 0., 0.],
                                   [0., 0., atmp[0][0], atmp[0][1], 1., 0.],
                                   [0., 0., atmp[1][0], atmp[1][1], 0., 1.],
                                   [0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0.]])
        else:
            self.eso.A = np.array([[0., 0., 1., 0., 0., 0.],
                                   [0., 0., 0., 1., 0., 0.],
                                   [0., 0., 0., 0., 1., 0.],
                                   [0., 0., 0., 0., 0., 1.],
                                   [0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0.]])

        self.eso.B = np.array([[0., 0.], [0., 0.], [b[0][0], b[0][1]], [b[1][0], b[1][1]], [0., 0.], [0., 0.]])

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        ### TODO implement centralized ADRFLC
        self.update_params(x, q_d)
        estimation = self.eso.get_state()

        q_t = x[:2]
        q_t_dot = x[2:4]

        K_d = [[1., 0.], [0., 1.]]
        K_p = [[1., 0.], [0., 1.]]

        v = q_d_ddot + self.K_d.dot(q_d_dot - q_t_dot) + self.K_p.dot(q_d - q_t)
        M = self.model.M(x)
        C = self.model.C(x)
        u = M.dot(v - estimation[3:5]) + C.dot(estimation[1:3])

        self.eso.update(x, u)

        return u
