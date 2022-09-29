import numpy as np


def Betz(rho_air, d, u):
    A = np.pi/4*d**2
    return 16/27 * 0.5 * rho_air * A * u**3

def Cp(rho_air, d, u):
    A = np.pi / 4 * d ** 2
    return 0.5 * rho_air * A * u ** 3