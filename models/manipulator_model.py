import numpy as np


class ManiuplatorModel:
    def __init__(self, Tp, m3, R3):
        self.Tp = Tp
        self.l1 = 0.5
        self.r1 = 0.04
        self.m1 = 3.
        self.l2 = 0.4
        self.r2 = 0.04
        self.m2 = 2.4
        self.I_1 = 1 / 12 * self.m1 * (3 * self.r1 ** 2 + self.l1 ** 2)
        self.I_2 = 1 / 12 * self.m2 * (3 * self.r2 ** 2 + self.l2 ** 2)
        self.m3 = m3
        self.r3 = R3
        self.I_3 = 2. / 5 * self.m3 * self.r3 ** 2

    def M(self, x):
        """
        Please implement the calculation of the mass matrix, according to the model derived in the exercise
        (2DoF planar manipulator with the object at the tip)
        """
        q1, q2, q1_dot, q2_dot = x

        d1 = self.l1 / 2
        d2 = self.l2 / 2

        alpha = self.m1 * d1**2 + self.I_1 + self.m2 * (self.l1**2 + d2**2) + self.I_2 +\
                self.m3 * (self.l1**2 + self.l2**2) + self.I_3
        beta = self.m2 * self.l1 * d2 + self.m3 * self.l1 * self.l2
        gamma = self.m2 * d2**2 + self.I_2 + self.m3 * self.l2**2 + self.I_3

        m1_1 = alpha + 2 * beta * np.cos(q2)
        m1_2 = gamma + beta * np.cos(q2)
        m2_1 = gamma + beta * np.cos(q2)
        m2_2 = gamma

        return np.array([[m1_1, m1_2], [m2_1, m2_2]])

    def C(self, x):
        """
        Please implement the calculation of the Coriolis and centrifugal forces matrix, according to the model derived
        in the exercise (2DoF planar manipulator with the object at the tip)
        """
        q1, q2, q1_dot, q2_dot = x

        d1 = self.l1 / 2
        d2 = self.l2 / 2

        alpha = self.m1 * d1**2 + self.I_1 + self.m2 * (self.l1**2 + d2**2) + self.I_2 +\
                self.m3 * (self.l1**2 + self.l2**2) + self.I_3
        beta = self.m2 * self.l1 * d2 + self.m3 * self.l1 * self.l2
        gamma = self.m2 * d2**2 + self.I_2 + self.m3 * self.l2**2 + self.I_3

        c1_1 = -beta * np.sin(q2) * q2_dot
        c1_2 = -beta * np.sin(q2) * (q1_dot + q2_dot)
        c2_1 = beta * np.sin(q2) * q1_dot
        c2_2 = 0

        return np.array([[c1_1, c1_2], [c2_1, c2_2]])
