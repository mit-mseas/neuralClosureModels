from IPython.core.debugger import set_trace

import numpy as np
import scipy.interpolate
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
import os
    
### Define a custom plotting function
class custom_plot:

    def __init__(self, true_y, y_no_nn, z, t, figsave_dir, args):
        self.true_y = true_y
        self.y_no_nn = y_no_nn
        self.t = t
        self.args = args
        self.figsave_dir = figsave_dir
        self.T, self.Z = np.meshgrid(t.numpy(), z.numpy())
        self.z = z
        
    def plot(self, *pred_y, epoch = 0):
        
        fig = plt.figure(figsize=(21, 5), facecolor='white')
        ax_N = fig.add_subplot(131)
        ax_P = fig.add_subplot(132)
        ax_Z = fig.add_subplot(133)
        
        self.plot_indiv(ax_N, 'True N (NNPZD)', tf.transpose(tf.squeeze(self.true_y[:, :, 0:self.args.nz], axis=1)).numpy())
        self.plot_indiv(ax_P, 'True P (NNPZD)', tf.transpose(tf.squeeze(self.true_y[:, :, self.args.nz:2*self.args.nz], axis=1)).numpy())
        self.plot_indiv(ax_Z, 'True Z (NNPZD)', tf.transpose(tf.squeeze(self.true_y[:, :, 2*self.args.nz:3*self.args.nz], axis=1)).numpy())

        fig = plt.figure(figsize=(21, 5), facecolor='white')
        ax_N = fig.add_subplot(131)
        ax_P = fig.add_subplot(132)
        ax_Z = fig.add_subplot(133)
        
        self.plot_indiv(ax_N, '|Difference N (no NN)|', tf.transpose(tf.squeeze(tf.abs(self.true_y - self.y_no_nn)[:, :, 0:self.args.nz], axis=1)).numpy())
        self.plot_indiv(ax_P, '|Difference P (no NN)|', tf.transpose(tf.squeeze(tf.abs(self.true_y - self.y_no_nn)[:, :, self.args.nz:2*self.args.nz], axis=1)).numpy())
        self.plot_indiv(ax_Z, '|Difference Z (no NN)|', tf.transpose(tf.squeeze(tf.abs(self.true_y - self.y_no_nn)[:, :, 2*self.args.nz:3*self.args.nz], axis=1)).numpy())
        
        if epoch != 0 or self.args.restart == 1 :
            fig = plt.figure(figsize=(21, 5), facecolor='white')
            ax_N = fig.add_subplot(131)
            ax_P = fig.add_subplot(132)
            ax_Z = fig.add_subplot(133)

            self.plot_indiv(ax_N, 'N (Learned)', tf.transpose(tf.squeeze(pred_y[0][:, :, 0:self.args.nz], axis=1)).numpy())
            self.plot_indiv(ax_P, 'P (Learned)', tf.transpose(tf.squeeze(pred_y[0][:, :, self.args.nz:2*self.args.nz], axis=1)).numpy())
            self.plot_indiv(ax_Z, 'Z (Learned)', tf.transpose(tf.squeeze(pred_y[0][:, :, 2*self.args.nz:3*self.args.nz], axis=1)).numpy())
            
            fig = plt.figure(figsize=(21, 5), facecolor='white')
            ax_N = fig.add_subplot(131)
            ax_P = fig.add_subplot(132)
            ax_Z = fig.add_subplot(133)
            
            self.plot_indiv(ax_N, '|Difference N (with NN)|', tf.transpose(tf.squeeze(tf.abs(self.true_y - pred_y[0][:, :, :self.args.state_dim])[:, :, 0:self.args.nz], axis=1)).numpy())
            self.plot_indiv(ax_P, '|Difference P (with NN)|', tf.transpose(tf.squeeze(tf.abs(self.true_y - pred_y[0][:, :, :self.args.state_dim])[:, :, self.args.nz:2*self.args.nz], axis=1)).numpy())
            self.plot_indiv(ax_Z, '|Difference Z (with NN)|', tf.transpose(tf.squeeze(tf.abs(self.true_y - pred_y[0][:, :, :self.args.state_dim])[:, :, 2*self.args.nz:3*self.args.nz], axis=1)).numpy())
            
        plt.show()
        
        if epoch != 0: 
            fig.savefig(os.path.join(self.figsave_dir, 'img'+str(epoch)))
        
    def plot_indiv(self, ax, title, B):
        ax.cla()
        ax.set_title(title, fontsize=14)
        ax.set_ylabel('z', fontsize=14)
        ax.set_xlabel('t (days)', fontsize=14)
        plot = ax.contourf(self.T, self.Z, B, cmap=cm.coolwarm,
                           antialiased=False, levels=np.linspace(0, np.max(B.flatten()), 20), extend='min')
        ax.set_ylim(self.z[-1], self.z[0])
        ax.set_xlim(self.t[0], self.t[-1])
        plt.colorbar(plot, ax=ax, shrink=0.5, aspect=10)
        
            
    def plot_npz(self, B):
        fig = plt.figure(figsize=(14, 10), facecolor='white')
        ax_N = fig.add_subplot(221)
        ax_P = fig.add_subplot(222)
        ax_Z = fig.add_subplot(223)
        
        self.plot_indiv(ax_N, 'Nutrients', tf.transpose(tf.squeeze(B[:, :, 0:self.args.nz], axis=1)).numpy())
        self.plot_indiv(ax_P, 'Phytoplankton', tf.transpose(tf.squeeze(B[:, :, self.args.nz:2*self.args.nz], axis=1)).numpy())
        self.plot_indiv(ax_Z, 'Zooplankton', tf.transpose(tf.squeeze(B[:, :, 2*self.args.nz:3*self.args.nz], axis=1)).numpy())

        plt.show()
        
    def plot_nnpzd(self, B):
        fig = plt.figure(figsize=(14, 20), facecolor='white')
        ax_NO3 = fig.add_subplot(321)
        ax_NH4 = fig.add_subplot(322)
        ax_P = fig.add_subplot(323)
        ax_Z = fig.add_subplot(324)
        ax_D = fig.add_subplot(325)
        
        self.plot_indiv(ax_NO3, 'Nitrates', tf.transpose(tf.squeeze(B[:, :, 0:self.args.nz], axis=1)).numpy())
        self.plot_indiv(ax_NH4, 'Ammonia', tf.transpose(tf.squeeze(B[:, :, self.args.nz:2*self.args.nz], axis=1)).numpy())
        self.plot_indiv(ax_P, 'Phytoplankton', tf.transpose(tf.squeeze(B[:, :, 2*self.args.nz:3*self.args.nz], axis=1)).numpy())
        self.plot_indiv(ax_Z, 'Zooplankton', tf.transpose(tf.squeeze(B[:, :, 3*self.args.nz:4*self.args.nz], axis=1)).numpy())
        self.plot_indiv(ax_D, 'Detritus', tf.transpose(tf.squeeze(B[:, :, 4*self.args.nz:5*self.args.nz], axis=1)).numpy())

        plt.show()
            
### Initial Conditions
class initial_cond:

    def __init__(self, app):
        self.app = app

    def __call__(self, t):

        if self.app.bio_model == 'NPZ':
            x0 = [self.app.T_bio - 0.05*2*self.app.T_bio, 0.05 * self.app.T_bio, 0.05 * self.app.T_bio]
        elif self.app.bio_model == 'NNPZD':
            x0 = [self.app.T_bio/2., self.app.T_bio/2. - 3*0.05*self.app.T_bio, 0.05*self.app.T_bio, 0.05*self.app.T_bio, 0.05*self.app.T_bio]
        return tf.expand_dims(tf.concat(x0, axis=0), axis=0)
    
    
### Create the Diffusion Coefficient function
class diff_coeff():
    
    def __init__(self, args, M, t, kind = 'linear'):
        self.args = args
        self.M_intrp = scipy.interpolate.interp1d(
            t,  # X
            M,  # Y
            kind=kind, axis=0,
            bounds_error=False,
            fill_value="extrapolate")
        
    def __call__(self, z, t):
        
        Kz = self.args.K_zb + ((self.args.K_z0 - self.args.K_zb) * \
        (np.arctan(self.args.gamma_K * (-(self.M_intrp(t) - z))) \
         - np.arctan(self.args.gamma_K * (-(self.M_intrp(t) - self.args.z_max)))) \
                              / (np.arctan(self.args.gamma_K * (-self.M_intrp(t))) \
                                 - np.arctan(self.args.gamma_K * (-(self.M_intrp(t) - self.args.z_max)))))
        
        return Kz