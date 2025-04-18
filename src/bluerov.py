import numpy as np
from numpy import cos, sin, pi, abs
from numba import njit
from src.parameters import (
    m, X_ud, Y_vd, Z_wd, I_zz, N_rd,
    X_u, X_uc, Y_v, Y_vc, Z_w, Z_wc,
    N_r, N_rc, g, F_bouy
)

@njit
def ssa(angle):
    return angle - 2*pi*np.floor_divide(angle + pi, 2*pi)

@njit
def bluerov_compute(t, x_, u_):
    ''' x_ = [x, y, z, psi, u, v, w, r] '''
    ''' u_ = [X, Y, Z, M_x, M_y, M_z] '''
    
    ''' State time derivative '''
    eta = np.array([x_[0], x_[1], x_[2], x_[3]])
    x, y, z, psi = eta
    psi = ssa(psi)
    nu = np.array([x_[4], x_[5], x_[6], x_[7]])
    u, v, w, r = nu
    tau = np.array([u_[0], u_[1], u_[2], u_[3]])
    X, Y, Z, M_z = tau
    
    x_d = cos(psi)*u - sin(psi)*v
    y_d = sin(psi)*u + cos(psi)*v
    z_d = w
    psi_d = r
    eta_dot = np.array([x_d, y_d, z_d, psi_d])
    
    u_d = 1 / (m - X_ud) * (X + (m - Y_vd) * v * r + (X_u + X_uc * abs(u)) * u) # experiments was carried out with error on the sign of Y_vd - should have been negative but was positive in the experiments    
    v_d = 1 / (m - Y_vd) * (Y - (m - X_ud) * u  * r + (Y_v + Y_vc * abs(v)) * v) # experiments was carried out with error on the sign of X_ud - should have been negative but was positive in the experiments  
    w_d = 1 / (m - Z_wd) * (Z + (Z_w + Z_wc * abs(w)) * w + m * g - F_bouy)
    r_d = 1 / (I_zz - N_rd) * (M_z - (X_ud - Y_vd) * u * v + (N_r + N_rc * abs(r)) * r)
    
    nu_dot = np.array([u_d, v_d, w_d, r_d])
    x_dot = np.hstack((eta_dot, nu_dot)) 
    return x_dot

def bluerov(t, x_, u_, params):
    x_dot = bluerov_compute(t, x_, u_)
    return x_dot
