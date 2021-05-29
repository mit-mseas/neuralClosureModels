
from src.utilities.DDE_Solver import ddeinttf 

import quadpy

import time
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

## Class to store user-defined variables
class arguments:
    def __init__(self, data_size = 1000, batch_time = 12, batch_time_skip = 2, batch_size = 5, epochs = 500, learning_rate = 0.05, decay_rate = 0.95, test_freq = 1, plot_freq = 2, 
                 d_max = 1.1, rnn_nmax = 1, rnn_dt = 0.5, state_dim = 2, adj_data_size = 2,
                 model_dir = 'DDE_runs/model_dir_test', restart = 0, val_percentage = 0.2, isplot = True, is_tstart_zero = True):
        self.data_size = data_size # Time-steps for which the ODE is solved for final loss computation and plotting
        self.batch_time = batch_time # Number of time-steps to skip for the batch final time. Total time interval will be dependent on the value of data_size as well
        self.batch_time_skip = batch_time_skip # Frequency of data points in the loss function
        self.batch_size = batch_size # Batch size 
        self.niters = np.ceil(self.data_size/(self.batch_time*self.batch_size) + 1).astype(int) # Number of iterations per epoch
        self.epochs = epochs # Number of epochs
        self.learning_rate = learning_rate # Initial learning rate
        self.decay_rate = decay_rate # parameter for exponential decay of learning rate
        self.test_freq = test_freq # Plotting frequency
        self.plot_freq = plot_freq
        self.isplot = isplot
        self.is_tstart_zero = is_tstart_zero
        
        self.d_max = d_max # The maximum value of time-delay present in the model
        
        self.rnn_nmax = rnn_nmax
        self.rnn_dt = rnn_dt
        
        self.state_dim = state_dim # State vector dimension for the ODE
        self.adj_data_size = adj_data_size # For how many time-steps to solve for the adjoint equation. Determined the accuracy of the gradients
        
        self.model_dir = model_dir
        self.basedir = os.getcwd()
        
        self.restart = restart
        
        self.val_percentage = val_percentage # Percentage of Length of time of training to be used for validation 

        self.base_str = 'cdefghijklmnopqrstuvwxyz' # A helper string


## To create batches
class create_batch:
    def __init__(self, true_y, true_y0, t, args):
        self.t = t
        self.true_y = true_y
        self.true_y0 = true_y0
        self.args = args

    def get_batch(self, flag = 0):

        s = np.random.choice(
            np.arange(self.args.data_size - self.args.batch_time,
                    dtype=np.int64), self.args.batch_size,
                    replace=False)
        
        dt = self.t[1] - self.t[0]
        n_back = np.ceil(np.abs(self.args.d_max/dt.numpy())).astype(int) 
        temp_y = self.true_y.numpy()

        if flag == 0:
            batch_size = self.args.batch_size
            batch_time = self.args.batch_time
        else:
            batch_size = self.args.batch_size
            batch_time = self.args.batch_time
            s = np.asarray([i for i in range(batch_size)])

        batch_y0 = [[] for _ in np.arange(batch_size)]
        
        for i in np.arange(batch_size):
            t_back = np.linspace(self.t[s[i]] - ((n_back-1)*dt.numpy()), self.t[s[i]], n_back)
            n_back_neg = t_back[t_back<np.min(self.t.numpy())].shape[0]
            for k in np.arange(n_back_neg):
                batch_y0[i].append(tf.expand_dims(self.true_y0(t_back[k]), axis=0))
            batch_y0[i].append(self.true_y[s[i] - (n_back - n_back_neg)+1:s[i]+1, ])
            batch_y0[i] = tf.expand_dims(tf.concat(batch_y0[i], axis=0), axis=1)

        batch_y0 = tf.concat(batch_y0, axis=1)
            
        batch_t0 = tf.linspace(self.t[0] - (n_back*dt.numpy()), self.t[0], n_back)
        batch_t = self.t[:batch_time+1:self.args.batch_time_skip]  # (T)
        batch_y = tf.stack([temp_y[s + i] for i in range(0, batch_time+1, self.args.batch_time_skip)], axis=0)  # (T, M, D)
        batch_t_start = self.t.numpy()[s]
        
        return tf.squeeze(batch_y0, axis=-2), batch_t0, batch_t, tf.squeeze(batch_y, axis=-2), batch_t_start


## Helper class to create interpolation functions
class create_interpolator():

    def __init__(self, batch_y0, batch_t0):
        self.batch_size = batch_y0.numpy().shape
        self.interpolator = scipy.interpolate.interp1d(
            batch_t0.numpy(),  # X
            np.reshape(batch_y0.numpy(), [self.batch_size[0], -1]),  # Y
            kind="linear", axis=0,
            bounds_error=False,
            fill_value="extrapolate"
        )

    def __call__(self, t):
        return  tf.convert_to_tensor(np.reshape(self.interpolator(t), self.batch_size[1:]), tf.float32)

class create_dirac_interpolator(create_interpolator): # dirac at end of the time interval

    def __init__(self, batch_y0, batch_t0, add_jump_val, jump_t):
        create_interpolator.__init__(self, batch_y0, batch_t0)
        self.add_jump_val = add_jump_val
        self.jump_t = jump_t

    def __call__(self, t):
        return self.add_jump_val + create_interpolator.__call__(self, t) if t==self.jump_t else create_interpolator.__call__(self, t)

class add_interpolator():

    def __init__(self, ini_interp, t_first, t_last):
        self.add_interp = [ini_interp]
        self.t = np.array([t_first, t_last])
        self.flag = False if (t_last - t_first <=0) else True

    def add(self, interp, t_first, t_last):
        self.add_interp.append(interp)
        self.t[-1] = t_first
        self.t = np.append(self.t, np.array([t_last]))

    def __call__(self, t):
        idx = np.digitize(t, self.t, right=self.flag) - 1
        return self.add_interp[idx](t)


## Define the adjoint for Discrete DDE
class adj_eqn:

    def __init__(self, model, args, jac = None):
        self.model = model
        self.args = args
        self.rom = model.rom_model
        self.jac = jac
        
    @tf.function
    def calc_jac(self, input, channels_to_add):
        with tf.GradientTape() as tape:
            tape.watch(input)
            h_x = self.model.call_nn_part(input, channels_to_add)

        dh_dx_i = tape.batch_jacobian(h_x, tf.reshape(input, [input.shape[0], input.shape[1], -1]))
        
        return dh_dx_i
    
    @tf.function
    def calc_rom_jac(self, input, t, t_start):
        with tf.GradientTape() as tape:
            tape.watch(input)
            rom_x = self.rom.rhs(input, t, t_start)

        drom_dx = tape.batch_jacobian(rom_x, input)
        return drom_dx
    
    def __call__(self, y, t, d, x, batch_size, t_start = np.array([0.])):

        l_dh_dx = tf.zeros([batch_size, self.args.state_dim], tf.float64)

        for k in np.arange(d[0]):
            
            input, channels_to_add = self.model.process_input(x, t + k*d[1], d, t_start)

            dh_dx_i = self.calc_jac(input, channels_to_add)
            
            l_dh_dx = l_dh_dx - tf.einsum('ab, abc -> ac', tf.cast(y(t + k*d[1]), tf.float64), tf.cast(dh_dx_i[:, :, d[0] - k - 1, ], tf.float64))
        
        input = x(t)
        
        if self.jac is not None:
            drom_dx = self.jac(input, t, t_start)
            
        else:
            drom_dx = self.calc_rom_jac(input, t, t_start)
        
        l_dh_dx = l_dh_dx - tf.einsum('ab, abc -> ac', tf.cast(y(t), tf.float64), tf.cast(drom_dx, tf.float64))
        
        return l_dh_dx
    
## Define the adjoint for ODE
class adj_eqn_ODE:

    def __init__(self, model, args, jac = None):
        self.model = model
        self.args = args
        self.rom = model.rom_model
        self.jac = jac
    
    @tf.function
    def calc_jac(self, input, channels_to_add):
        with tf.GradientTape() as tape:
            tape.watch(input)
            h_x = self.model.main(input, channels_to_add)

        dh_dx_i = tape.batch_jacobian(h_x, tf.reshape(input, [input.shape[0], -1]))
        
        return dh_dx_i
    
    @tf.function
    def calc_rom_jac(self, input, t, t_start):
        with tf.GradientTape() as tape:
            tape.watch(input)
            rom_x = self.rom.rhs(input, t, t_start)

        drom_dx = tape.batch_jacobian(rom_x, input)
        

    def __call__(self, y, t, d, x, batch_size, t_start = np.array([0.])):

        l_dh_dx = tf.zeros([batch_size, self.args.state_dim], tf.float64)
            
        input, channels_to_add = self.model.process_input(x, t, d, t_start)

        dh_dx_i = self.calc_jac(input, channels_to_add)
        
        l_dh_dx = l_dh_dx - tf.einsum('ab, abc -> ac', tf.cast(y(t), tf.float64), tf.cast(dh_dx_i, tf.float64))
        
        input = x(t)
        
        if self.jac is not None:
            drom_dx = self.jac(input, t, t_start)
            
        else:
            drom_dx = self.calc_rom_jac(input, t, t_start)
        
        l_dh_dx = l_dh_dx - tf.einsum('ab, abc -> ac', tf.cast(y(t), tf.float64), tf.cast(drom_dx, tf.float64))
        
        return l_dh_dx


## Function to compute gradients w.r.t. trainable variables
class grad_train_var:
    
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
        self.out_shape = list(self.y(t_lowerlim).shape)
        self.scheme = quadpy.c1.gauss_legendre(5)
        self.grad = self.integrate_lam_dfdp()
        
    @tf.function
    def calc_jac(self, input, channels_to_add):
        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_weights)
            h = self.model.call_nn_part(input, channels_to_add)

        dh_dp = tape.jacobian(h, self.model.trainable_weights, experimental_use_pfor=False)
        
        return dh_dp
        

    def lam_dfdp(self, t):

        input, channels_to_add = self.model.process_input(self.y, t, self.d, self.t_start)
        
        dh_dp = self.calc_jac(input, channels_to_add)
        
        dh_dp  = tf.concat([tf.reshape(dh_dp[i], self.out_shape + [-1]) for i in range(len(dh_dp))], axis=-1)
    
        var_shape_len = len(tf.shape(dh_dp).numpy())
      
        lam_dh_dp =  - tf.einsum('ab,ab'+self.args.base_str[:var_shape_len-2] + 
                               '->a'+self.args.base_str[:var_shape_len-2],tf.cast(self.lam(t), tf.float64) , 
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

## Main training class
class train_nDDE:
    
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

    def train(self, true_y, true_y0, t, val_true_y):
        end = time.time()
        for epoch in range(1, self.args.epochs + 1):
            
            if epoch == 1:

                pred_y = ddeinttf(self.func, true_y0, tf.concat([t, self.val_obj.val_t], axis=0), fargs=(self.d,), alg_name = self.args.ode_alg_name, nsteps = self.args.nsteps)
                
                pred_y_train, pred_y_val = self.val_obj.data_split(pred_y)
                loss_train = tf.squeeze(self.loss(true_y, pred_y_train)) 
                
                loss_val = tf.squeeze(self.loss(val_true_y, pred_y_val))
                
                print('Epoch {:04d} | Train Loss {:.6f} | Val Loss {:.6f} | Time Elapsed {:.4f}'.format(epoch - 1, loss_train.numpy(), loss_val.numpy(), self.time_meter.avg/60), 'mins')
                
                self.loss_history.add(loss_train, loss_val, epoch - 1)
                self.loss_history.save()
                
                if self.args.isplot == True:
                    self.plot_obj.plot(pred_y, epoch = epoch)
                
            for itr in tqdm(range(1, self.args.niters + 1), desc = 'Iterations', file=sys.stdout):

                if itr == 1:
                    batch_y0, batch_t0, batch_t, batch_y, batch_t_start = self.batch_obj.get_batch(1)
                    batch_size = self.args.batch_size
                else:
                    batch_y0, batch_t0, batch_t, batch_y, batch_t_start = self.batch_obj.get_batch()
                    batch_size = self.args.batch_size

                if self.args.is_tstart_zero: batch_t_start = tf.zeros(list(batch_t_start.shape))
                    
                dloss_dpred_y_base = tf.zeros([batch_size, self.args.state_dim], tf.float64)
                pred_adj = []
                batch_adj_t = []

                pred_y = ddeinttf(self.func, create_interpolator(batch_y0, batch_t0), batch_t, fargs=(self.d, batch_t_start), alg_name = self.args.ode_alg_name, nsteps = self.args.nsteps)

                grads_avg = [tf.zeros(self.func.trainable_weights[k].shape, tf.float32) for k in range(len(self.func.trainable_weights))]

                #create a interpolator function from the forward pass
                pred_y0_y_interp = create_interpolator(tf.concat([batch_y0, pred_y], axis=0), tf.concat([batch_t0, batch_t], axis=0))
                
                pred_y = pred_y[1:, ]
                batch_y = batch_y[1:, ]
                
                with tf.GradientTape() as tape:
                    tape.watch(pred_y)
                    loss = self.loss(batch_y, pred_y)
                dloss_dpred_y_whole = tape.gradient(loss, pred_y)
                
                pred_adj_intrvl = tf.zeros([self.args.adj_data_size, batch_size, self.args.state_dim], tf.float64)
                batch_adj_t_intrvl = tf.linspace(batch_t[-1] + self.args.d_max, batch_t[-1], self.args.adj_data_size)

                for k in range(len(batch_t)-1, 0, -1):
                    dloss_dpred_y = dloss_dpred_y_whole[k-1, ]
                    
                    # Initial conditions for augmented adjoint ODE
                    if k == len(batch_t)-1:
                        adj_eqn_ic_interp = add_interpolator(create_dirac_interpolator(pred_adj_intrvl, batch_adj_t_intrvl, dloss_dpred_y, batch_adj_t_intrvl[-1]), batch_adj_t_intrvl[0], batch_adj_t_intrvl[-1])
                    else:
                        adj_eqn_ic_interp.add(create_dirac_interpolator(pred_adj_intrvl, batch_adj_t_intrvl, dloss_dpred_y, batch_adj_t_intrvl[-1]), batch_adj_t_intrvl[0], batch_adj_t_intrvl[-1])

                    batch_adj_t_intrvl = tf.linspace(batch_t[k], batch_t[k-1], self.args.adj_data_size)

                    # Solve for the augmented adjoint ODE 
                    pred_adj_intrvl = ddeinttf(self.adj_func, adj_eqn_ic_interp, batch_adj_t_intrvl, fargs = (self.d, pred_y0_y_interp, batch_size, batch_t_start), alg_name = self.args.ode_alg_name, nsteps = self.args.nsteps)
                    
                    # Compute gradients w.r.t. trainable weights

                    grads = grad_train_var(self.func, create_interpolator(pred_adj_intrvl, batch_adj_t_intrvl), pred_y0_y_interp, batch_adj_t_intrvl[-1], 
                                        batch_adj_t_intrvl[0], self.d, self.args, batch_t_start).grad
                    
                    grads_avg_intrvl = [tf.cast(-tf.reduce_mean(grad_indiv, axis = 0), tf.float32) for grad_indiv in grads]

                    for i in range(len(self.func.trainable_weights)):
                        grads_avg[i] = grads_avg[i] + grads_avg_intrvl[i]
                
                grads_zip = zip(grads_avg, self.func.trainable_weights)

                self.optimizer.apply_gradients(grads_zip)

            self.time_meter.update(time.time() - end)

            if epoch % self.args.test_freq == 0:

                pred_y = ddeinttf(self.func, true_y0, tf.concat([t, self.val_obj.val_t], axis=0), fargs=(self.d,), alg_name = self.args.ode_alg_name, nsteps = self.args.nsteps)
            
                pred_y_train, pred_y_val = self.val_obj.data_split(pred_y)
                loss_train = tf.squeeze(self.loss(true_y, pred_y_train)) #tf.reduce_mean(tf.keras.losses.MSE(pred_y, true_y)) 
                
                loss_val = tf.squeeze(self.loss(val_true_y, pred_y_val))
            
                self.loss_history.add(loss_train, loss_val, epoch)
                self.loss_history.save()
            
                print('Epoch {:04d} | Train Loss {:.6f} | Val Loss {:.6f} | LR {:.4f}| Time Elapsed {:.4f}'.format(epoch, loss_train.numpy(), loss_val.numpy(), self.optimizer.learning_rate(self.optimizer.iterations.numpy() - 1).numpy(), self.time_meter.avg/60), 'mins')

                self.func.save_weights(self.checkpoint_dir.format(epoch=epoch))
                
            if (epoch % self.args.plot_freq == 0) and self.args.isplot == True:
                self.plot_obj.plot(pred_y, epoch = epoch)

            end = time.time()


## Helper class to compute average time, etc.
class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
        
### Helper class to store, write and read loss values
class history:
    def __init__(self, args):
        self.train_loss = []
        self.val_loss = []
        self.epoch = []
        self.args = args

    def add(self, train_loss, val_loss, epoch):
        self.train_loss.append(train_loss.numpy())
        self.val_loss.append(val_loss.numpy())
        self.epoch.append(epoch)

    def save(self):
        with open(self.args.model_dir + '/loss_history.p', 'wb') as f:
            pickle.dump([self.epoch, self.train_loss, self.val_loss], f)

    def read(self):
        with open(self.args.model_dir + '/loss_history.p', 'rb') as f:
            [self.epoch, self.train_loss, self.val_loss] = pickle.load(f)
            
### Create validation set
class create_validation_set:
    def __init__(self, y0, t, args):
        dt = t[1] - t[0]
        n_back = np.ceil(np.abs(args.d_max/dt.numpy())).astype(int)
        t0 = tf.linspace(t[0] - args.d_max, t[0], n_back)
        y0_t0 = []
        
        for i in range(t0.shape[0]):
            y0_t0.append(y0(t0[i]))

        self.val_true_y0 = add_interpolator(create_interpolator(tf.concat(y0_t0, axis=0), t0), t0[0], t0[-1])
        val_t_len =  args.val_percentage * (t[-1] - t[0])
        n_val = np.ceil(np.abs(val_t_len/dt.numpy())).astype(int)
        self.val_t = tf.linspace(t[-1], t[-1] + val_t_len, n_val)
        self.t = t
        self.args = args
        
    def data_split(self, pred_y_whole):
        pred_y_train = pred_y_whole[0:len(self.t), :, :self.args.state_dim]
        pred_y_val = pred_y_whole[len(self.t):, :, :self.args.state_dim]
        return pred_y_train, pred_y_val
    
    def data_split_any(self, pred_y_whole):
        pred_y_train = pred_y_whole[0:len(self.t), ]
        pred_y_val = pred_y_whole[len(self.t):, :, ]
        return pred_y_train, pred_y_val

#     def data(self, func, y, t, d = None):
#         self.val_true_y0.add(create_interpolator(y, t), t[0], t[-1])
#         if d != None:
#             val_true_y = ddeinttf(func, self.val_true_y0, self.val_t, fargs=(d,), alg_name = self.args.ode_alg_name, nsteps = self.args.nsteps)
#         else:
#             val_true_y = ddeinttf(func, self.val_true_y0, self.val_t, alg_name = self.args.ode_alg_name, nsteps = self.args.nsteps)
#         return val_true_y


    
