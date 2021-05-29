from IPython.core.debugger import set_trace

import numpy as np
from math import *
import tensorflow as tf

tf.keras.backend.set_floatx('float32')

#### User-defined parameters
class bio_eqn_args:
    def __init__(self, T, nt, z, k_w, alpha, V_m, I_0, K_u, Psi, Xi, R_m, Lambda, gamma, Tau, Phi, Omega, T_bio, bio_model):
        self.T = T
        self.nt = nt
        self.dt = self.T / self.nt

        self.z = z
        self.k_w = k_w
        self.alpha = alpha
        self.V_m = V_m
        self.I_0 = I_0
        self.K_u = K_u
        self.Psi = Psi
        self.Xi = Xi
        self.R_m = R_m
        self.Lambda = Lambda
        self.gamma = gamma
        self.Tau = Tau
        self.Phi = Phi
        self.Omega = Omega
        self.T_bio = T_bio

        I = I_0 * exp(k_w * z)
        f_z = (alpha * I) / sqrt(V_m**2 + alpha**2 * I**2)
        self.G = V_m * f_z 

        self.bio_model = bio_model

### RHS of CDV Eqns
class bio_eqn:

    def __init__(self, app):
        self.app = app
        
    def rhs(self, x_t, t, t_start):

        if self.app.bio_model == 'NNPZD':
            dxdt = [self.app.Omega * x_t[:, 1] - tf.math.multiply(self.app.G * tf.math.truediv(tf.math.multiply(x_t[:, 0], tf.math.exp(-self.app.Psi * x_t[:, 1])), (self.app.K_u + x_t[:, 0])), x_t[:, 2])]
            
            dxdt.append(self.app.Phi * x_t[:, 4] + self.app.Tau * x_t[:, 3] - tf.math.multiply(self.app.G * tf.math.truediv(x_t[:, 1], (self.app.K_u + x_t[:, 1])), x_t[:, 2]) - self.app.Omega * x_t[:, 1])

            dxdt.append(self.app.G * tf.math.multiply(tf.math.truediv(tf.math.multiply(x_t[:, 0], tf.math.exp(-self.app.Psi * x_t[:, 1])), (self.app.K_u + x_t[:, 0])) + tf.math.truediv(x_t[:, 1], (self.app.K_u + x_t[:, 1])), x_t[:, 2]) 
                - tf.math.multiply(self.app.R_m * (1 - tf.math.exp(-self.app.Lambda * x_t[:, 2])), x_t[:, 3]) - self.app.Xi * x_t[:, 2])

            dxdt.append((1 - self.app.gamma) * self.app.R_m * tf.math.multiply((1 - tf.math.exp(-self.app.Lambda * x_t[:, 2])), x_t[:, 3]) - self.app.Tau * x_t[:, 3])

            dxdt.append(self.app.gamma * self.app.R_m * tf.math.multiply((1 - tf.math.exp(-self.app.Lambda * x_t[:, 2])), x_t[:, 3]) + self.app.Xi * x_t[:, 2] - self.app.Phi * x_t[:, 4])

        elif self.app.bio_model == 'NPZD':
            dxdt = [self.app.Phi * x_t[:, 3] + self.app.Tau * x_t[:, 2] - tf.math.multiply(self.app.G * tf.math.truediv(x_t[:, 0], (self.app.K_u + x_t[:, 0])), x_t[:, 1])]

            dxdt.append(tf.math.multiply(self.app.G * tf.math.truediv(x_t[:, 0], (self.app.K_u + x_t[:, 0])), x_t[:, 1]) - tf.math.multiply(self.app.R_m * (1 - tf.math.exp(-self.app.Lambda * x_t[:, 1])), x_t[:, 2]) - self.app.Xi * x_t[:, 1])

            dxdt.append((1 - self.app.gamma) * tf.math.multiply(self.app.R_m * (1 - tf.math.exp(-self.app.Lambda * x_t[:, 1])), x_t[:, 2]) - self.app.Tau * x_t[:, 2])

            dxdt.append(self.app.gamma * tf.math.multiply(self.app.R_m * (1 -tf.math.exp(-self.app.Lambda * x_t[:, 1])), x_t[:, 2]) + self.app.Xi * x_t[:, 1] - self.app.Phi * x_t[:, 3])

        elif self.app.bio_model == 'NPZ':
            dxdt = [self.app.Xi * x_t[:, 1] + self.app.Tau * x_t[:, 2] - tf.math.multiply(self.app.G * tf.math.truediv(x_t[:, 0], (self.app.K_u + x_t[:, 0])), x_t[:, 1]) 
                + self.app.gamma * tf.math.multiply(self.app.R_m * (1 - tf.math.exp(-self.app.Lambda * x_t[:, 1])), x_t[:, 2])]

            dxdt.append(tf.math.multiply(self.app.G * tf.math.truediv(x_t[:, 0], (self.app.K_u + x_t[:, 0])), x_t[:, 1]) - tf.math.multiply(self.app.R_m * (1 - tf.math.exp(-self.app.Lambda * x_t[:, 1])), x_t[:, 2]) - self.app.Xi * x_t[:, 1])

            dxdt.append((1 - self.app.gamma) * tf.math.multiply(self.app.R_m * (1 - tf.math.exp(-self.app.Lambda * x_t[:, 1])), x_t[:, 2]) - self.app.Tau * x_t[:, 2])

        dxdt = tf.stack(dxdt, axis=1)

        return dxdt
    
    def __call__(self, x, t, t_start = np.array([0.])):
        x_t = x(t)
        
        return self.rhs(x_t, t, t_start)
    
    def jac_npz(self, x_t, t, t_start):
        
        dS_NdN = [- self.app.G * self.app.K_u * tf.math.truediv(x_t[:, 1], (self.app.K_u + x_t[:, 0])**2)]
        dS_NdP = [- self.app.G * tf.math.truediv(x_t[:, 0], (self.app.K_u + x_t[:, 0])) + self.app.Xi + self.app.gamma * tf.math.multiply(self.app.R_m * self.app.Lambda * tf.math.exp(-self.app.Lambda * x_t[:, 1]), x_t[:, 2])]
        dS_NdZ = [self.app.Tau + self.app.gamma * self.app.R_m * (1 - tf.math.exp(-self.app.Lambda * x_t[:, 1]))]
        
        dS_PdN = [self.app.G * self.app.K_u * tf.math.truediv(x_t[:, 1], (self.app.K_u + x_t[:, 0])**2)]
        dS_PdP = [self.app.G * tf.math.truediv(x_t[:, 0], (self.app.K_u + x_t[:, 0])) - self.app.Xi - tf.math.multiply(self.app.R_m * self.app.Lambda * tf.math.exp(-self.app.Lambda * x_t[:, 1]), x_t[:, 2])]
        dS_PdZ = [ - self.app.R_m * (1 - tf.math.exp(-self.app.Lambda * x_t[:, 1]))]
        
        dS_ZdN = [tf.zeros(x_t[:, 1].shape)]
        dS_ZdP = [(1 - self.app.gamma) * tf.math.multiply(self.app.R_m * self.app.Lambda * tf.math.exp(-self.app.Lambda * x_t[:, 1]), x_t[:, 2])]
        dS_ZdZ = [- self.app.Tau + (1 - self.app.gamma) * self.app.R_m * (1 - tf.math.exp(-self.app.Lambda * x_t[:, 1]))]
        
        dS_dN = tf.stack(dS_NdN + dS_PdN + dS_ZdN, axis=1)
        dS_dP = tf.stack(dS_NdP + dS_PdP + dS_ZdP, axis=1)
        dS_dZ = tf.stack(dS_NdZ + dS_PdZ + dS_ZdZ, axis=1)
        
        jac = tf.stack([dS_dN, dS_dP, dS_dZ], axis=-1)
        
        return jac
        
def convert_high_complex_to_low_complex_states(x_high_complex, args): 
    true_x_low_complex = []
    if args.bio_model_high_complex == 'NNPZD':
        true_x_low_complex.append(x_high_complex[:, :, 0] + x_high_complex[:, :, 1] + x_high_complex[:, :, 4])
        true_x_low_complex.append(x_high_complex[:, :, 2])
        true_x_low_complex.append(x_high_complex[:, :, 3])

    elif args.bio_model_high_complex == 'NPZD':
        true_x_low_complex.append(x_high_complex[:, :, 0] + x_high_complex[:, :, 3])
        true_x_low_complex.append(x_high_complex[:, :, 1])
        true_x_low_complex.append(x_high_complex[:, :, 2])

    true_x_low_complex = tf.stack(true_x_low_complex, axis=-1)

    return true_x_low_complex
