from IPython.core.debugger import set_trace

import numpy as np
from math import *
import tensorflow as tf

tf.keras.backend.set_floatx('float32')

#### User-defined parameters
class bio_eqn_args:
    def __init__(self, T, nt, nz, z_max, k_w, alpha, V_m, I_0, K_u, Psi, Xi, R_m, Lambda, gamma, Tau, Phi, 
                 Omega, T_bio_min, T_bio_max, wp, wd, bio_model, K_zb = 8.64, K_z0 = 864., gamma_K = 0.1, T_mld = 100):
        self.T = T
        self.nt = nt
        self.nz = nz
        self.dt = self.T / self.nt

        self.z_max = z_max
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
        self.T_bio_min = T_bio_min
        self.T_bio_max = T_bio_max 
        self.wp = wp
        self.wd = wd
        
        self.K_zb = K_zb
        self.K_z0 = K_z0
        self.gamma_K = gamma_K
        self.T_mld = T_mld
        
        self.z = tf.linspace(0., z_max, nz)
        self.dz = self.z[1] - self.z[0]
        self.T_bio = tf.linspace(T_bio_min, T_bio_max, nz)
        
        I_0_t = lambda t: I_0 - 50.*np.sin((t + 365./6.) * 2. * pi / 365.)
#         I_0_t = lambda t: I_0 - 50.*np.sin((t + 365./4.) * 2. * pi / 365.)
        
        self.I = lambda t: I_0_t(t) * tf.math.exp(k_w * self.z)
#         I = lambda t: I_0_t(t) * tf.math.exp(k_w * tf.linspace(-25., -25., nz))
        f_z = lambda t: tf.math.truediv((alpha * self.I(t)), tf.math.sqrt(V_m**2 + alpha**2 * self.I(t)**2))
        self.G = lambda t: tf.cast(V_m * tf.stack([f_z(t[i]) for i in range(t.shape[0])], axis=0), tf.float32)

        self.bio_model = bio_model

### RHS of Bio Eqns
class bio_eqn:

    def __init__(self, app, diff_coeff):
        self.app = app
        self.diff_coeff = diff_coeff
        
    def diff_mat(self, t):
        
        diff_mat_batch = []
        for k in range(len(t)):
            K_zt = self.diff_coeff(self.app.z, t[k])
            
            main_mat_indices = []
            main_mat_values = []
            
            indices = np.concatenate([np.expand_dims(np.arange(1, self.app.nz-1), axis=-1), np.expand_dims(np.arange(1, self.app.nz-1), axis=-1)], axis=-1).astype(int).tolist()
            
            main_mat_indices += indices
            main_mat_values += list(K_zt[1:self.app.nz-1] * 2.)
            
            indices = np.concatenate([np.expand_dims(np.arange(1, self.app.nz-1), axis=-1), np.expand_dims(np.arange(1-1, self.app.nz-1-1), axis=-1)], axis=-1).astype(int).tolist()
            
            main_mat_indices += indices
            main_mat_values += list(- K_zt[1:self.app.nz-1] + (K_zt[1-1:self.app.nz-1-1] - K_zt[1+1:self.app.nz-1+1]))
            
            indices = np.concatenate([np.expand_dims(np.arange(1, self.app.nz-1), axis=-1), np.expand_dims(np.arange(1+1, self.app.nz-1+1), axis=-1)], axis=-1).astype(int).tolist()
            
            main_mat_indices += indices
            main_mat_values += list(- K_zt[1:self.app.nz-1] - (K_zt[1-1:self.app.nz-1-1] - K_zt[1+1:self.app.nz-1+1]))
            
            i = 0 
            main_mat_indices.append([i, i]) 
            main_mat_values.append(K_zt[i] * 2.)

            main_mat_indices.append([i, i+1]) 
            main_mat_values.append( - K_zt[i] * 2.)

            i = self.app.nz-1 
            main_mat_indices.append([i, i]) 
            main_mat_values.append(K_zt[i] * 2.)

            main_mat_indices.append([i, i-1]) 
            main_mat_values.append( - K_zt[i] * 2.)
            
            main_mat_indices = np.array(main_mat_indices)
            
            if self.app.bio_model == 'NPZ':
                
                main_mat_indices = np.vstack((main_mat_indices, main_mat_indices + self.app.nz, main_mat_indices + 2*self.app.nz))
                main_mat_values = main_mat_values * 3
                
                diff_mat = (1./self.app.dz**2) * tf.cast(tf.sparse.to_dense(tf.sparse.SparseTensor(main_mat_indices.tolist(), main_mat_values, dense_shape=[self.app.nz*3, self.app.nz*3]), validate_indices=False), tf.float32)
                
            elif self.app.bio_model == 'NNPZD':
                
                main_mat_indices = np.vstack((main_mat_indices, main_mat_indices + self.app.nz, main_mat_indices + 2*self.app.nz, main_mat_indices + 3*self.app.nz, main_mat_indices + 4*self.app.nz))
                main_mat_values = main_mat_values * 5
                
                diff_mat = (1./self.app.dz**2) * tf.cast(tf.sparse.to_dense(tf.sparse.SparseTensor(main_mat_indices.tolist(), main_mat_values, dense_shape=[self.app.nz*5, self.app.nz*5]), validate_indices=False), tf.float32)
            
            diff_mat_batch.append(diff_mat)

        diff_mat_batch = tf.stack(diff_mat_batch, axis=0)
        
        return diff_mat_batch
        
    def rhs(self, x_t, t, t_start):
        t = t + t_start
        G = self.app.G(t)
#         M_t = tf.transpose(tf.convert_to_tensor([self.diff_coeff.M_intrp(t)], tf.float32))
#         M_t = tf.tile(M_t, [1, self.app.nz])
       
        if self.app.bio_model == 'NNPZD':

            dxdt = [self.app.Omega * x_t[:, self.app.nz:2*self.app.nz] \
                    - tf.math.multiply(G * tf.math.truediv(tf.math.multiply(x_t[:, 0:self.app.nz], tf.math.exp(-self.app.Psi * x_t[:, self.app.nz:2*self.app.nz])), (self.app.K_u + x_t[:, 0:self.app.nz])), x_t[:, 2*self.app.nz:3*self.app.nz])]
            
            dxdt.append(self.app.Phi * x_t[:, 4*self.app.nz:5*self.app.nz] 
                        + self.app.Tau * x_t[:, 3*self.app.nz:4*self.app.nz] 
                        - tf.math.multiply(G * tf.math.truediv(x_t[:, self.app.nz:2*self.app.nz], 
                                                                        (self.app.K_u + x_t[:, self.app.nz:2*self.app.nz])), x_t[:, 2*self.app.nz:3*self.app.nz]) - self.app.Omega * x_t[:, self.app.nz:2*self.app.nz])

            dxdt.append(G * tf.math.multiply(tf.math.truediv(tf.math.multiply(x_t[:, 0:1*self.app.nz], tf.math.exp(-self.app.Psi * x_t[:, self.app.nz:2*self.app.nz])), (self.app.K_u + x_t[:, 0:1*self.app.nz])) + tf.math.truediv(x_t[:, self.app.nz:2*self.app.nz], (self.app.K_u + x_t[:, self.app.nz:2*self.app.nz])), x_t[:, 2*self.app.nz:3*self.app.nz]) 
                - tf.math.multiply(self.app.R_m * (1 - tf.math.exp(-self.app.Lambda * x_t[:, 2*self.app.nz:3*self.app.nz])), x_t[:, 3*self.app.nz:4*self.app.nz]) - self.app.Xi * x_t[:, 2*self.app.nz:3*self.app.nz])

            dxdt.append((1 - self.app.gamma) * self.app.R_m * tf.math.multiply((1 - tf.math.exp(-self.app.Lambda * x_t[:, 2*self.app.nz:3*self.app.nz])), x_t[:, 3*self.app.nz:4*self.app.nz]) - self.app.Tau * x_t[:, 3*self.app.nz:4*self.app.nz])

            dxdt.append(self.app.gamma * self.app.R_m * tf.math.multiply((1 - tf.math.exp(-self.app.Lambda * x_t[:, 2*self.app.nz:3*self.app.nz])), x_t[:, 3*self.app.nz:4*self.app.nz]) + self.app.Xi * x_t[:, 2*self.app.nz:3*self.app.nz] - self.app.Phi * x_t[:, 4*self.app.nz:5*self.app.nz])# + self.app.wd * tf.truediv(x_t[:, 4*self.app.nz:5*self.app.nz], M_t))

        elif self.app.bio_model == 'NPZ':
            dxdt = [self.app.Xi * x_t[:, 1*self.app.nz:2*self.app.nz] + self.app.Tau * x_t[:, 2*self.app.nz:3*self.app.nz] - tf.math.multiply(G * tf.math.truediv(x_t[:, 0*self.app.nz:1*self.app.nz], (self.app.K_u + x_t[:, 0*self.app.nz:1*self.app.nz])), x_t[:, 1*self.app.nz:2*self.app.nz]) 
                + self.app.gamma * tf.math.multiply(self.app.R_m * (1 - tf.math.exp(-self.app.Lambda * x_t[:, 1*self.app.nz:2*self.app.nz])), x_t[:, 2*self.app.nz:3*self.app.nz])]

            dxdt.append(tf.math.multiply(G * tf.math.truediv(x_t[:, 0*self.app.nz:1*self.app.nz], (self.app.K_u + x_t[:, 0*self.app.nz:1*self.app.nz])), x_t[:, 1*self.app.nz:2*self.app.nz]) - tf.math.multiply(self.app.R_m * (1 - tf.math.exp(-self.app.Lambda * x_t[:, 1*self.app.nz:2*self.app.nz])), x_t[:, 2*self.app.nz:3*self.app.nz]) - self.app.Xi * x_t[:, 1*self.app.nz:2*self.app.nz])# + self.app.wp * tf.truediv(x_t[:, 1*self.app.nz:2*self.app.nz], M_t))

            dxdt.append((1 - self.app.gamma) * tf.math.multiply(self.app.R_m * (1 - tf.math.exp(-self.app.Lambda * x_t[:, 1*self.app.nz:2*self.app.nz])), x_t[:, 2*self.app.nz:3*self.app.nz]) - self.app.Tau * x_t[:, 2*self.app.nz:3*self.app.nz])
        
        diff_term = tf.cast(tf.einsum('abc, ac -> ab', tf.cast(self.diff_mat(t), tf.float64), tf.cast(x_t, tf.float64)), tf.float32)
        
        dxdt = tf.concat(dxdt, axis=-1) - diff_term
        
        return dxdt
    
    def __call__(self, x, t, t_start = np.array([0.])):
        x_t = x(t)
        
        return self.rhs(x_t, t, t_start)
    
    def jac_npz(self, x_t, t, t_start):
        t = t + t_start
        G = self.app.G(t)
#         M_t = tf.transpose(tf.convert_to_tensor([self.diff_coeff.M_intrp(t)], tf.float32))
#         M_t = tf.tile(M_t, [1, self.app.nz])
        
        dS_NdN = [- G * self.app.K_u * tf.math.truediv(x_t[:, 1*self.app.nz:2*self.app.nz], (self.app.K_u + x_t[:, 0*self.app.nz:1*self.app.nz])**2)]
        dS_NdP = [- G * tf.math.truediv(x_t[:, 0*self.app.nz:1*self.app.nz], (self.app.K_u + x_t[:, 0*self.app.nz:1*self.app.nz])) + self.app.Xi + self.app.gamma * tf.math.multiply(self.app.R_m * self.app.Lambda * tf.math.exp(-self.app.Lambda * x_t[:, 1*self.app.nz:2*self.app.nz]), x_t[:, 2*self.app.nz:3*self.app.nz])]
        dS_NdZ = [self.app.Tau + self.app.gamma * self.app.R_m * (1 - tf.math.exp(-self.app.Lambda * x_t[:, 1*self.app.nz:2*self.app.nz]))]
        
        dS_PdN = [G * self.app.K_u * tf.math.truediv(x_t[:, 1*self.app.nz:2*self.app.nz], (self.app.K_u + x_t[:, 0*self.app.nz:1*self.app.nz])**2)]
        dS_PdP = [G * tf.math.truediv(x_t[:, 0*self.app.nz:1*self.app.nz], (self.app.K_u + x_t[:, 0*self.app.nz:1*self.app.nz])) - self.app.Xi - tf.math.multiply(self.app.R_m * self.app.Lambda * tf.math.exp(-self.app.Lambda * x_t[:, 1*self.app.nz:2*self.app.nz]), x_t[:, 2*self.app.nz:3*self.app.nz])]# + (self.app.wp / M_t)]
        dS_PdZ = [- self.app.R_m * (1 - tf.math.exp(-self.app.Lambda * x_t[:, 1*self.app.nz:2*self.app.nz]))]
        
        dS_ZdN = [tf.zeros(x_t[:, 1*self.app.nz:2*self.app.nz].shape)]
        dS_ZdP = [(1 - self.app.gamma) * tf.math.multiply(self.app.R_m * self.app.Lambda * tf.math.exp(-self.app.Lambda * x_t[:, 1*self.app.nz:2*self.app.nz]), x_t[:, 2*self.app.nz:3*self.app.nz])]
        dS_ZdZ = [- self.app.Tau + (1 - self.app.gamma) * self.app.R_m * (1 - tf.math.exp(-self.app.Lambda * x_t[:, 1*self.app.nz:2*self.app.nz]))]
        
        dS_dN = tf.concat([tf.linalg.diag(dS_NdN[0])] + [tf.linalg.diag(dS_PdN[0])] + [tf.linalg.diag(dS_ZdN[0])], axis=1)
        dS_dP = tf.concat([tf.linalg.diag(dS_NdP[0])] + [tf.linalg.diag(dS_PdP[0])]  + [tf.linalg.diag(dS_ZdP[0])] , axis=1)
        dS_dZ = tf.concat([tf.linalg.diag(dS_NdZ[0])]  + [tf.linalg.diag(dS_PdZ[0])] + [tf.linalg.diag(dS_ZdZ[0])], axis=1)
        
        jac = tf.concat([dS_dN, dS_dP, dS_dZ], axis=-1) - self.diff_mat(t)
        
        return jac
        
def convert_high_complex_to_low_complex_states(x_high_complex, app): 
    true_x_low_complex = []

    true_x_low_complex.append(x_high_complex[:, :, 0*app.nz:1*app.nz] 
                              + x_high_complex[:, :, 1*app.nz:2*app.nz] 
                              + x_high_complex[:, :, 4*app.nz:5*app.nz])
    true_x_low_complex.append(x_high_complex[:, :, 2*app.nz:3*app.nz])
    true_x_low_complex.append(x_high_complex[:, :, 3*app.nz:4*app.nz])

    true_x_low_complex = tf.concat(true_x_low_complex, axis=-1)

    return true_x_low_complex