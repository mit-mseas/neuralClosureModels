from IPython.core.debugger import set_trace

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

tf.keras.backend.set_floatx('float32')

## Define some useful classes
        
### Create modes using the FOM solution
class create_mean_modes:
    def __init__(self, fom_sol, app, t):
        self.fom_sol = fom_sol
        self.app = app
        self.t = t

    def __call__(self):

        u_analy = tf.transpose(self.fom_sol)

        u_mean = tf.expand_dims(tf.reduce_mean(u_analy, axis=-1), axis=1)

        S, U, V = tf.linalg.svd(u_analy - tf.tile(u_mean, [1, len(self.t)]))

        ui = U[:, 0:self.app.rom_dims]

        return u_mean, ui
   
    
### Define a custom plotting function
class custom_plot:

    def __init__(self, true_y, true_y_red_ic, y_no_nn, t, figsave_dir, args):
        self.true_y = true_y
        self.true_y_red_ic = true_y_red_ic
        self.y_no_nn = y_no_nn
        self.t = t
        self.figsave_dir = figsave_dir
        self.args = args
        self.colors = ['b', 'g', 'r', 'k', 'c', 'm']

    def plot(self, *pred_y, epoch = 0):
        fig = plt.figure(figsize=(6, 4), facecolor='white')
        ax = fig.add_subplot(111)

        ax.cla()
        ax.set_title('Trajectories')
        ax.set_xlabel('t')
        ax.set_ylabel('$a_i$')
        ax.set_xlim(min(self.t.numpy()), max(self.t.numpy()))
        ax.set_ylim(-1.25, 1.25)

        for i in range(self.args.rom_dims):
#             ax.plot(self.t.numpy(), self.true_y.numpy()[:, 0, i], self.colors[i % self.args.rom_dims]+'-', label = 'True mode '+str(i+1))
            ax.plot(self.t.numpy(), self.true_y_red_ic.numpy()[:, 0, i], self.colors[i % self.args.rom_dims]+'-', label = 'True-Red. IC mode '+str(i+1))
            ax.plot(self.t.numpy(), self.y_no_nn.numpy()[:, 0, i], self.colors[i % self.args.rom_dims]+'-.', label = 'GP mode '+str(i+1))
            if epoch != 0 or self.args.restart == 1 :
                ax.plot(self.t.numpy(), pred_y[0].numpy()[:, 0, i], self.colors[i % self.args.rom_dims]+'--', label = 'Learned mode '+str(i+1))

        ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.show() 

        if epoch != 0: 
            fig.savefig(os.path.join(self.figsave_dir, 'img'+str(epoch)))
            
            
### Initial Conditions
class initial_cond:

    def __init__(self, x, app):
        self.x = x
        self.app = app

    def __call__(self, t):
        u0 = self.x / (1. + np.sqrt(1./self.app.t0) * np.exp(self.app.Re * self.x**2 / 4., dtype = np.float64))
        return tf.convert_to_tensor([u0], dtype=tf.float32)
    
class red_initial_cond:
    
    def __init__(self, ai_t0, u_mean, u_modes):
        self.ai_t0 = ai_t0
        self.u_mean = u_mean
        self.u_modes = u_modes
        
    def __call__(self, t):

        u0_rom = tf.transpose(self.u_mean, perm=[1, 0]) \
            + tf.cast(tf.einsum('ab, db -> da', tf.cast(self.u_modes, tf.float64), tf.cast(self.ai_t0(t), tf.float64)), tf.float32)
        
        return u0_rom
    

            
            
