from src.utilities.DDE_Solver import ddeinttf
from src.solvers.neuralDDE_with_adjoint_accel import *

import quadpy

import time
import timeit
import sys
import os
from tqdm import tqdm
from IPython.core.debugger import set_trace

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import integrate
import scipy.interpolate
import pickle

tf.keras.backend.set_floatx('float32')

#### Class for user-defined variables
class nddde_arguments(arguments):
    def __init__(self, data_size = 1000, batch_time = 12, batch_time_skip = 2, batch_size = 5, epochs = 500, learning_rate = 0.05, decay_rate = 0.95, test_freq = 1, plot_freq = 2, 
                 d_max = 1., nn_d1 = 0., nn_d2 = 0.5, state_dim = 2, adj_data_size = 2,
                 model_dir = 'DistDDE_runs/model_dir_test', restart = 0, val_percentage = 0.2, isplot = True, is_tstart_zero = True):
        
        arguments.__init__(self, data_size = data_size, batch_time = batch_time, batch_time_skip = batch_time_skip, batch_size = batch_size, epochs = epochs, 
                           learning_rate = learning_rate, decay_rate = decay_rate, test_freq = test_freq, plot_freq = plot_freq, 
                            d_max = d_max, state_dim = state_dim, adj_data_size = adj_data_size,
                            model_dir = model_dir, restart = restart, val_percentage = val_percentage, isplot = isplot, is_tstart_zero = is_tstart_zero)
        
        self.nn_d1 = nn_d1
        self.nn_d2 = nn_d2

#### Class to construct initial conditions for coupled DDE form
class process_DistDDE_IC:

    def __init__(self, z, aux_model, t_lowerlim, t_upperlim):
        self.z = z
        self.g = aux_model
        self.t_ll = t_lowerlim
        self.t_ul = t_upperlim
        self.scheme = quadpy.c1.gauss_legendre(5)
        self.intgral_z = self.integrate_z()

    def convert_to_numpy(self, t):
        return np.stack([self.g(self.z(t[i])).numpy() for i in range(len(t))], axis=-1)

    def integrate_z(self):
        val = self.scheme.integrate(self.convert_to_numpy, [self.t_ll, self.t_ul])
        return tf.convert_to_tensor(val, tf.float32)

    def __call__(self, t):
        if tf.rank(t) == 0: 
            return tf.concat([self.z(t), self.intgral_z], axis=-1)
        else : 
            return tf.stack([tf.concat([self.z(t), self.intgral_z], axis=-1) for t in t], axis=0)


#### Adjoint system
class nddde_adj_eqn:

    def __init__(self, model, args, jac = None):
        self.f = model.main
        self.g = model.aux
        self.rom = model.rom_model
        self.args = args
        self.jac = jac

    @tf.function
    def calc_jac_main(self, z_y_in):
        with tf.GradientTape() as tape:
            tape.watch(z_y_in)
            f_zy = self.f(z_y_in)

        df_dzy = tape.batch_jacobian(f_zy, z_y_in)
        return df_dzy
    
    @tf.function
    def calc_jac_aux(self, z):
        with tf.GradientTape() as tape:
            tape.watch(z)
            g_z = self.g(z)

        dg_dz = tape.batch_jacobian(g_z, z)
        return dg_dz
    
    @tf.function
    def calc_rom_jac(self, input, t, t_start):
        with tf.GradientTape() as tape:
            tape.watch(input)
            rom_x = self.rom.rhs(input, t, t_start)

        drom_dx = tape.batch_jacobian(rom_x, input)
    
    def __call__(self, lam_mu, t, d, z_y, t_start):
        
        z_y_in = z_y(t)
        z = z_y_in[:, :self.args.state_dim]
        
        df_dzy = self.calc_jac_main(z_y_in)
        
        dg_dz = self.calc_jac_aux(z)
        
        if self.jac is not None:
            drom_dz = self.jac(z, t, t_start)
            
        else:
            drom_dz = self.calc_rom_jac(z, t, t_start)
      
        lam_rhs = - tf.einsum('ab, abc -> ac', tf.cast(lam_mu(t)[:, :self.args.state_dim], tf.float64), tf.cast(df_dzy[:, :, :self.args.state_dim] + drom_dz, tf.float64)) \
                    - tf.einsum('ab, abc -> ac', tf.cast(lam_mu(t + d[0])[:, self.args.state_dim:], tf.float64), tf.cast(dg_dz, tf.float64)) \
                      + tf.einsum('ab, abc -> ac', tf.cast(lam_mu(t + d[1])[:, self.args.state_dim:], tf.float64), tf.cast(dg_dz, tf.float64))

        mu_rhs = - tf.einsum('ab, abc -> ac', tf.cast(lam_mu(t)[:, :self.args.state_dim], tf.float64), tf.cast(df_dzy[:, :, self.args.state_dim:], tf.float64))

        return tf.concat([lam_rhs, mu_rhs], axis=-1)


#### Class for validation set
class create_validation_set_nddde(create_validation_set):
    def __init__(self, y0, t, args):
        create_validation_set.__init__(self, y0, t, args)
        self.args = args


## Function to compute gradients w.r.t. trainable variables
class grad_train_var_nddde:

    def __init__(self, model, lam, y, t_lowerlim, t_upperlim, d, args, t_start = np.array([0.])):
        self.model = model
        self.lam = lam
        self.y = y
        self.t_ll = t_lowerlim
        self.t_ul = t_upperlim
        self.d = d
        self.args = args
        self.t_start = t_start
        self.weight_shapes = [self.model.trainable_weights[i].shape for i in range(len(self.model.trainable_weights))]
        self.out_shape = list(self.lam.shape)
        self.scheme = quadpy.c1.gauss_legendre(5)
        self.grad = self.integrate_lam_dfdp()
    
    @tf.function
    def calc_jac(self, input):
        
        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_weights)
            h = self.model(input)
 
        dh_dp = tape.jacobian(h, self.model.trainable_weights, experimental_use_pfor=False)

        return dh_dp
        
    def lam_dfdp(self, t):

        input = self.y(t)
        
        dh_dp = self.calc_jac(input)

        dh_dp  =tf.concat([tf.reshape(dh_dp[i], self.out_shape + [-1]) for i in range(len(dh_dp))], axis=-1)
    
        var_shape_len = len(tf.shape(dh_dp).numpy())
      
        lam_dh_dp =  - tf.einsum('ab,ab'+self.args.base_str[:var_shape_len-2] + 
                               '->a'+self.args.base_str[:var_shape_len-2],tf.cast(self.lam, tf.float64) , 
                                                        tf.cast(dh_dp, tf.float64)).numpy()

        return lam_dh_dp

    def stack_lam_dfdp(self, t):

        return np.stack([self.lam_dfdp(t[i]) for i in range(len(t))], axis=-1)

    def integrate_lam_dfdp(self):
        
        lam_dh_dp = self.scheme.integrate(self.stack_lam_dfdp, [self.t_ll, self.t_ul])
        
        m = 0
        n = 0
        lam_dh_dp_list = []
        for i in range(len(self.weight_shapes)):
            n += tf.math.reduce_prod(self.weight_shapes[i]).numpy()
            lam_dh_dp_list.append(tf.reshape(lam_dh_dp[:, m:n], [-1] + list(self.weight_shapes[i])))
            m = n
       
        return lam_dh_dp_list



##### Training class
class train_nDistDDE:

    def __init__(self, func, adj_func, d, loss_obj, batch_obj, optimizer, args, plot_obj, time_meter, checkpoint_dir, validation_obj, loss_history_obj):
        self.func = func
        self.adj_func = adj_func
        self.d = d
        self.loss = loss_obj
        self.batch_obj = batch_obj
        self.optimizer = optimizer
        self.args = args
        self.plot_obj = plot_obj
        self.time_meter = time_meter
        self.checkpoint_dir = checkpoint_dir
        self.val_obj = validation_obj
        self.loss_history = loss_history_obj

    def train(self, true_z, true_z0, t, val_true_z):
        end = time.time()
        for epoch in range(1, self.args.epochs + 1):

            if epoch == 1:
                
                process_true_z0 = process_DistDDE_IC(true_z0, self.func.aux, t_lowerlim = t[0] - self.args.nn_d2, t_upperlim = t[0] - self.args.nn_d1)
                
                pred_zy = ddeinttf(self.func, process_true_z0, tf.concat([t, self.val_obj.val_t], axis=0), fargs=(self.d,), alg_name = self.args.ode_alg_name, nsteps = self.args.nsteps)
                
                pred_z_train, pred_z_val = self.val_obj.data_split(pred_zy)
                loss_train = tf.squeeze(self.loss(true_z, pred_z_train))  
                
                loss_val = tf.squeeze(self.loss(val_true_z, pred_z_val))
                
                print('Epoch {:04d} | Train Loss {:.6f} | Val Loss {:.6f} | Time Elapsed {:.4f}'.format(epoch - 1, loss_train.numpy(), loss_val.numpy(), self.time_meter.avg/60), 'mins')
                
                self.loss_history.add(loss_train, loss_val, epoch - 1)
                self.loss_history.save()

                if self.args.isplot == True:
                    self.plot_obj.plot(pred_zy, epoch = epoch)

            for itr in tqdm(range(1, self.args.niters + 1), desc = 'Iterations', file=sys.stdout):

                if itr == 1:
                    batch_z0, batch_t0, batch_t, batch_z, batch_t_start = self.batch_obj.get_batch(1)
                    batch_size = self.args.batch_size
                else:
                    batch_z0, batch_t0, batch_t, batch_z, batch_t_start = self.batch_obj.get_batch()
                    batch_size = self.args.batch_size
                
                if self.args.is_tstart_zero: batch_t_start = tf.zeros(list(batch_t_start.shape))
                    
                dloss_dpred_z_base = tf.zeros([batch_size, self.args.state_dim], tf.float64)
                pred_adj = []
                batch_adj_t = []

                batch_zy0 = process_DistDDE_IC(create_interpolator(batch_z0, batch_t0), self.func.aux, t_lowerlim = batch_t0[-1] - self.args.nn_d2, 
                                                        t_upperlim = batch_t0[-1] - self.args.nn_d1)
                
                pred_zy = ddeinttf(self.func, batch_zy0 , batch_t, fargs=([self.args.nn_d1, self.args.nn_d2], batch_t_start), alg_name = self.args.ode_alg_name, nsteps = self.args.nsteps)

                grads_avg = [tf.zeros(self.func.trainable_weights[k].shape, tf.float32) for k in range(len(self.func.trainable_weights))]

                #create a interpolator function from the forward pass
                pred_zy0_zy_interp = create_interpolator(tf.concat([batch_zy0(batch_t0), pred_zy], axis=0), tf.concat([batch_t0, batch_t], axis=0))
                
                pred_zy = pred_zy[1:, ]
                batch_z = batch_z[1:, ]
                pred_z = pred_zy[:, :, :self.args.state_dim]
                
                with tf.GradientTape() as tape:
                    tape.watch(pred_z)
                    loss = self.loss(batch_z, pred_z)
                dloss_dpred_z_whole = tape.gradient(loss, pred_z)

                pred_adj_intrvl = tf.zeros([self.args.adj_data_size, batch_size, pred_zy.shape[-1]], tf.float64)
                batch_adj_t_intrvl = tf.linspace(batch_t[-1] + self.args.d_max, batch_t[-1], self.args.adj_data_size)

                for k in range(len(batch_t)-1, 0, -1):
                    dloss_dpred_z = dloss_dpred_z_whole[k-1, ]

                    # Initial conditions for augmented adjoint ODE
                    if k == len(batch_t)-1:
                        adj_eqn_ic_interp = add_interpolator(create_dirac_interpolator(pred_adj_intrvl, batch_adj_t_intrvl, 
                                                                                    tf.concat([dloss_dpred_z, tf.zeros([batch_size, pred_zy.shape[-1] - self.args.state_dim])], axis=-1), 
                                                                                    batch_adj_t_intrvl[-1]), batch_adj_t_intrvl[0], batch_adj_t_intrvl[-1])
                    else:
                        adj_eqn_ic_interp.add(create_dirac_interpolator(pred_adj_intrvl, batch_adj_t_intrvl, 
                                                                        tf.concat([dloss_dpred_z, tf.zeros([batch_size, pred_zy.shape[-1] - self.args.state_dim])], axis=-1), 
                                                                        batch_adj_t_intrvl[-1]), batch_adj_t_intrvl[0], batch_adj_t_intrvl[-1])

                    batch_adj_t_intrvl = tf.linspace(batch_t[k], batch_t[k-1], self.args.adj_data_size)

                    # Solve for the augmented adjoint ODE 
                    pred_adj_intrvl = ddeinttf(self.adj_func, adj_eqn_ic_interp, batch_adj_t_intrvl, fargs = (self.d, pred_zy0_zy_interp, batch_t_start), alg_name = self.args.ode_alg_name, nsteps = self.args.nsteps)
                 
                    # Compute gradients w.r.t. trainable weights

                    grads = grad_train_var(self.func, create_interpolator(pred_adj_intrvl, batch_adj_t_intrvl), pred_zy0_zy_interp, batch_adj_t_intrvl[-1], 
                                        batch_adj_t_intrvl[0], self.d, self.args, batch_t_start).grad

                    grads_avg_intrvl = [tf.cast(-tf.reduce_mean(grad_indiv, axis = 0), tf.float32) for grad_indiv in grads]

                    for i in range(len(self.func.trainable_weights)):
                        grads_avg[i] = grads_avg[i] + grads_avg_intrvl[i]

                grads = grad_train_var_nddde(self.func.aux, pred_adj_intrvl[-1, :, self.args.state_dim:], create_interpolator(batch_z0, batch_t0), - self.args.nn_d2, 
                                        - self.args.nn_d1, self.d, self.args, batch_t_start).grad

                grads_avg_intrvl = [tf.cast(-tf.reduce_mean(grad_indiv, axis = 0), tf.float32) for grad_indiv in grads]
                
                for i in range(len(self.func.main.trainable_weights), len(self.func.trainable_weights)): # This assumes that the starting weights corresponds to main NN and the last ones to aux NN
                    grads_avg[i] = grads_avg[i] + grads_avg_intrvl[i - len(self.func.main.trainable_weights)]
                
                grads_zip = zip(grads_avg, self.func.trainable_weights)
                self.optimizer.apply_gradients(grads_zip)

            self.time_meter.update(time.time() - end)

            # Plotting
            if epoch % self.args.test_freq == 0:
                process_true_z0 = process_DistDDE_IC(true_z0, self.func.aux, t_lowerlim = t[0] - self.args.nn_d2, t_upperlim = t[0] - self.args.nn_d1)

                pred_zy = ddeinttf(self.func, process_true_z0, tf.concat([t, self.val_obj.val_t], axis=0), fargs=(self.d,), alg_name = self.args.ode_alg_name, nsteps = self.args.nsteps)
                
                pred_z_train, pred_z_val = self.val_obj.data_split(pred_zy)
                loss_train = tf.squeeze(self.loss(true_z, pred_z_train)) 
                
                loss_val = tf.squeeze(self.loss(val_true_z, pred_z_val))

                self.loss_history.add(loss_train, loss_val, epoch)
                self.loss_history.save()

                print('Epoch {:04d} | Train Loss {:.6f} | Val Loss {:.6f} | LR {:.4f} | Time Elapsed {:.4f}'.format(epoch, loss_train.numpy(), loss_val.numpy(), self.optimizer.learning_rate(self.optimizer.iterations.numpy() - 1).numpy(), self.time_meter.avg/60), 'mins')

                self.func.save_weights(self.checkpoint_dir.format(epoch=epoch))

            if (epoch % self.args.plot_freq == 0) and self.args.isplot == True:
                self.plot_obj.plot(pred_zy, epoch = epoch)
                

            end = time.time()