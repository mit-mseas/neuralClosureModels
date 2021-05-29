from src.utilities.DDE_Solver import ddeinttf 

from IPython.core.debugger import set_trace

import numpy as np
import tensorflow as tf

tf.keras.backend.set_floatx('float32')

#### User-defined parameters
class ad_eqn_args:
    def __init__(self, T, nt, L, nx, Re, u_bc_0, u_bc_L, multi_solve_size):
        self.T = T
        self.nt = nt
        self.dt = self.T / self.nt

        self.L = L
        self.nx = nx
        self.dx = self.L / self.nx

        self.Re = Re
        self.nu = 1./self.Re
        self.t0 = np.exp(self.Re/8, dtype = np.float64)

        self.u_bc_0 = u_bc_0
        self.u_bc_L = u_bc_L
        
        self.multi_solve_size = multi_solve_size # Batch size 


#### Class to create advection and diffusion matrices
class operators:

    def __init__(self, app):
        self.app = app
        self.diff()

    def diff(self):
        main_mat_indices = []
        main_mat_values = []
        bc_mat_indices = []
        bc_mat_values = []

        bc_mat_indices.append([0, 0])
        bc_mat_values.append(self.app.u_bc_0)
        bc_mat_indices.append([0, self.app.nx-1])
        bc_mat_values.append(self.app.u_bc_L)

        for i in range(self.app.nx):
            main_mat_indices.append([i, i])
            main_mat_values.append(-2.)
            if i != 0: 
                main_mat_indices.append([i, i-1])
                main_mat_values.append(1.)
            if i != self.app.nx-1: 
                main_mat_indices.append([i, i+1])
                main_mat_values.append(1.)

        self.diff_mat = (self.app.nu/self.app.dx**2) * tf.sparse.to_dense(tf.sparse.SparseTensor(main_mat_indices, 
                                                                                                 main_mat_values, dense_shape=[self.app.nx, self.app.nx]), validate_indices=False)
        self.diff_bc = (self.app.nu/self.app.dx**2) * tf.sparse.to_dense(tf.sparse.SparseTensor(bc_mat_indices, 
                                                                                                bc_mat_values, dense_shape=[1, self.app.nx]), validate_indices=False)

    def adv_uw(self, u):
        adv_mat_values = []
        adv_mat_indices = []
        adv_bc_values = []
        adv_bc_indices = []

        for i in range(u.shape[0]):
            u_sign = np.where(u[i, :] >= 0, 1, -1)
            indices = np.concatenate([np.expand_dims(np.repeat(i, self.app.nx), axis=-1), np.expand_dims(np.arange(self.app.nx), axis=-1), np.expand_dims(np.arange(self.app.nx), axis=-1)], axis=-1).astype(int).tolist()
            
            adv_mat_indices += indices
            adv_mat_values += list(u_sign * u[i, :] / self.app.dx)
            
            indices = np.concatenate([np.expand_dims(np.repeat(i, self.app.nx - 2), axis=-1), np.expand_dims(np.arange(1, self.app.nx - 1), axis=-1), np.expand_dims(np.arange(1, self.app.nx - 1) - u_sign[1:self.app.nx - 1], axis=-1)], axis=-1).astype(int).tolist()
            
            adv_mat_indices += indices
            
            adv_mat_values += list(- u_sign[1:self.app.nx - 1] * u[i, 1:self.app.nx - 1] / self.app.dx)
            
            adv_bc_indices.append([i, 0]) 
            adv_bc_values.append(- u[i, 0] * self.app.u_bc_0 / self.app.dx)
            adv_bc_indices.append([i, self.app.nx-1]) 
            adv_bc_values.append(u[i, self.app.nx-1] * self.app.u_bc_L / self.app.dx)
            
#             for j in range(self.app.nx):
#                 if u[i, j] >=0:
#                     adv_mat_indices.append([i, j, j]) 
#                     adv_mat_values.append(u[i, j]/self.app.dx)
#                     if j == 0:
#                         adv_bc_indices.append([i, j]) 
#                         adv_bc_values.append(- u[i, j] * self.app.u_bc_0 / self.app.dx)
#                     else:
#                         adv_mat_indices.append([i, j, j-1]) 
#                         adv_mat_values.append(- u[i, j]/self.app.dx)

#                 if u[i, j] < 0:
#                     adv_mat_indices.append([i, j, j]) 
#                     adv_mat_values.append(- u[i, j]/self.app.dx)
#                     if j == self.app.nx-1: 
#                         adv_bc_indices.append([i, j]) 
#                         adv_bc_values.append(u[i, j] * self.app.u_bc_L / self.app.dx)
#                     else:
#                         adv_mat_indices.append([i, j, j+1]) 
#                         adv_mat_values.append(u[i, j]/self.app.dx)

        adv_mat =  tf.sparse.to_dense(tf.sparse.SparseTensor(adv_mat_indices, adv_mat_values, dense_shape=[u.shape[0], self.app.nx, self.app.nx]), validate_indices=False) 
        
        if adv_bc_indices == []:
            adv_bc = tf.zeros([u.shape[0], self.app.nx], tf.float32)
        else:
            adv_bc =  tf.sparse.to_dense(tf.sparse.SparseTensor(adv_bc_indices, adv_bc_values, dense_shape=[u.shape[0], self.app.nx]), validate_indices=False)
    
        return adv_mat, adv_bc

    def adv_cd(self, u):
        adv_mat_values = []
        adv_mat_indices = []
        adv_bc_values = []
        adv_bc_indices = []

        for i in range(u.shape[0]):

            indices = np.concatenate([np.expand_dims(np.repeat(i, self.app.nx - 1), axis=-1), np.expand_dims(np.arange(0, self.app.nx - 1), axis=-1), np.expand_dims(np.arange(0, self.app.nx - 1) + 1, axis=-1)], axis=-1).astype(int).tolist()
            
            adv_mat_indices += indices
            adv_mat_values += list(u[i, 0:self.app.nx - 1] / (2.*self.app.dx))
            
            indices = np.concatenate([np.expand_dims(np.repeat(i, self.app.nx - 1), axis=-1), np.expand_dims(np.arange(1, self.app.nx), axis=-1), np.expand_dims(np.arange(1, self.app.nx) - 1, axis=-1)], axis=-1).astype(int).tolist()
            
            adv_mat_indices += indices
            
            adv_mat_values += list(- u[i, 1:self.app.nx] / (2.*self.app.dx))
            
            adv_bc_indices.append([i, 0]) 
            adv_bc_values.append(- u[i, 0] * self.app.u_bc_0 /(2.*self.app.dx))
            adv_bc_indices.append([i, self.app.nx-1]) 
            adv_bc_values.append(u[i, self.app.nx-1] * self.app.u_bc_L / (2.*self.app.dx))
            
#             for j in range(self.app.nx):

#                 if j == self.app.nx-1:
#                     adv_bc_indices.append([i, j]) 
#                     adv_bc_values.append( u[i, j] * self.app.u_bc_0 / (2. * self.app.dx))
#                 else:
#                     adv_mat_indices.append([i, j, j + 1]) 
#                     adv_mat_values.append(u[i, j]/(2. * self.app.dx))

#                 if j == 0:
#                     adv_bc_indices.append([i, j]) 
#                     adv_bc_values.append(- u[i, j] * self.app.u_bc_0 / (2. * self.app.dx))
#                 else:
#                     adv_mat_indices.append([i, j, j-1]) 
#                     adv_mat_values.append(- u[i, j]/(2. * self.app.dx))

        adv_mat =  tf.sparse.to_dense(tf.sparse.SparseTensor(adv_mat_indices, adv_mat_values, dense_shape=[u.shape[0], self.app.nx, self.app.nx]), validate_indices=False) 
        adv_bc =  tf.sparse.to_dense(tf.sparse.SparseTensor(adv_bc_indices, adv_bc_values, dense_shape=[u.shape[0], self.app.nx]), validate_indices=False)

        return adv_mat, adv_bc

#### R.H.S of du_dt
class ad_eqn:

    def __init__(self, op, app):
        self.op = op
        self.app = app
        
    def rhs(self, u_t, t, t_start):

        diff_term = tf.cast(tf.einsum('bc, ac -> ab', tf.cast(self.op.diff_mat, tf.float64), tf.cast(u_t, tf.float64)), tf.float32) + tf.tile(self.op.diff_bc, [u_t.shape[0], 1]) 

        adv_mat, adv_bc = self.op.adv_uw(u_t)
        adv_term = tf.cast(tf.einsum('abc, ac -> ab', tf.cast(adv_mat, tf.float64), tf.cast(u_t, tf.float64)), tf.float32) + adv_bc

        dudt = diff_term - adv_term

        return dudt
    
       
    def __call__(self, u, t, t_start = np.array([0.])):

        u_t = u(t)
        
        return self.rhs(u_t, t, t_start)
    
    def jac(self, u, t, t_start):
        
        jac_adv_mat_values = []
        jac_adv_mat_indices = []
        jac_adv_bc_values = []
        jac_adv_bc_indices = []

        for i in range(u.shape[0]):
            u_sign = np.where(u[i, :] >= 0, 1, -1)
            
            indices = np.concatenate([np.expand_dims(np.repeat(i, self.app.nx-2), axis=-1), np.expand_dims(np.arange(1, self.app.nx-1), axis=-1), np.expand_dims(np.arange(1, self.app.nx-1), axis=-1)], axis=-1).astype(int).tolist()
            
            jac_adv_mat_indices += indices
            jac_adv_mat_values += list( (u_sign[1:self.app.nx-1] * 2.*tf.gather(u[i, :], list(np.arange(1, self.app.nx-1).astype(int)), axis=-1) - u_sign[1:self.app.nx-1] * tf.gather(u[i, :], list(np.arange(1, self.app.nx-1).astype(int) - u_sign[1:self.app.nx-1]), axis=-1)) / self.app.dx)
            
            indices = np.concatenate([np.expand_dims(np.repeat(i, self.app.nx - 2), axis=-1), np.expand_dims(np.arange(1, self.app.nx - 1), axis=-1), np.expand_dims(np.arange(1, self.app.nx - 1) - u_sign[1:self.app.nx - 1], axis=-1)], axis=-1).astype(int).tolist()
            
            jac_adv_mat_indices += indices
            jac_adv_mat_values += list(- u_sign[1:self.app.nx - 1] * u[i, 1:self.app.nx - 1] / self.app.dx)
            
            j = 0
            jac_adv_mat_indices.append([i, j, j]) 
            jac_adv_mat_values.append((2. * u[i, j])/self.app.dx)
                    
            jac_adv_bc_indices.append([i, j, j]) 
            jac_adv_bc_values.append(- self.app.u_bc_0 / self.app.dx)
            
            j = self.app.nx-1
            jac_adv_mat_indices.append([i, j, j]) 
            jac_adv_mat_values.append((- 2 * u[i, j])/self.app.dx)
                    
            jac_adv_bc_indices.append([i, j, j]) 
            jac_adv_bc_values.append(self.app.u_bc_L / self.app.dx)
            
#             for j in range(self.app.nx):
#                 if u[i, j] >=0:
                   
#                     if j == 0:
#                         jac_adv_mat_indices.append([i, j, j]) 
#                         jac_adv_mat_values.append((2. * u[i, j])/self.app.dx)
                    
#                         jac_adv_bc_indices.append([i, j, j]) 
#                         jac_adv_bc_values.append(- self.app.u_bc_0 / self.app.dx)
#                     else:
#                         jac_adv_mat_indices.append([i, j, j]) 
#                         jac_adv_mat_values.append((2. * u[i, j] - u[i, j-1])/self.app.dx)
                    
#                         jac_adv_mat_indices.append([i, j, j-1]) 
#                         jac_adv_mat_values.append(- u[i, j]/self.app.dx)

#                 if u[i, j] < 0:
                   
#                     if j == self.app.nx-1: 
#                         jac_adv_mat_indices.append([i, j, j]) 
#                         jac_adv_mat_values.append((- 2 * u[i, j])/self.app.dx)
                    
#                         jac_adv_bc_indices.append([i, j, j]) 
#                         jac_adv_bc_values.append(self.app.u_bc_L / self.app.dx)
#                     else:
#                         jac_adv_mat_indices.append([i, j, j]) 
#                         jac_adv_mat_values.append((- 2 * u[i, j] + u[i, j+1])/self.app.dx)
                    
#                         jac_adv_mat_indices.append([i, j, j+1]) 
#                         jac_adv_mat_values.append(u[i, j]/self.app.dx)
                        
        jac_adv_mat =  tf.sparse.to_dense(tf.sparse.SparseTensor(jac_adv_mat_indices, jac_adv_mat_values, dense_shape=[u.shape[0], self.app.nx, self.app.nx]), validate_indices=False) 
        
        if jac_adv_bc_indices == []:
            jac_adv_bc = tf.zeros([u.shape[0], self.app.nx, self.app.nx], tf.float32)
        else:
            jac_adv_bc =  tf.sparse.to_dense(tf.sparse.SparseTensor(jac_adv_bc_indices, jac_adv_bc_values, dense_shape=[u.shape[0], self.app.nx, self.app.nx]), validate_indices=False)
            
        jac = self.op.diff_mat - jac_adv_mat - jac_adv_bc
            
        return jac
