from IPython.core.debugger import set_trace

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
    
### Define a custom plotting function
class custom_plot:

    def __init__(self, true_y, y_no_nn, t, figsave_dir, args):
        self.true_y = true_y
        self.y_no_nn = y_no_nn
        self.t = t
        self.figsave_dir = figsave_dir
        self.args = args
        self.colors = ['b', 'g', 'r', 'k', 'c', 'm']

    def plot(self, *pred_y, epoch = 0):
        fig = plt.figure(figsize=(6, 4), facecolor='white')
        ax_x1 = fig.add_subplot(111)

        ax_x1.cla()
        ax_x1.set_title('Bio Model Comparison')
        ax_x1.set_xlabel('t')
        ax_x1.set_ylabel('Bio Variable')
        ax_x1.plot(self.t.numpy(), self.true_y.numpy()[:, 0, 0], '-r', label = 'N (High Complex)')
        ax_x1.plot(self.t.numpy(), self.y_no_nn.numpy()[:, 0, 0], '--r', label = 'N (NPZ)')
        ax_x1.plot(self.t.numpy(), self.true_y.numpy()[:, 0, 1], '-g', label = 'P (High Complex)')
        ax_x1.plot(self.t.numpy(), self.y_no_nn.numpy()[:, 0, 1], '--g', label = 'P (NPZ)')
        ax_x1.plot(self.t.numpy(), self.true_y.numpy()[:, 0, 2], '-b', label = 'Z (High Complex)')
        ax_x1.plot(self.t.numpy(), self.y_no_nn.numpy()[:, 0, 2], '--b', label = 'Z (NPZ)')

        if epoch != 0 or self.args.restart == 1 :
            ax_x1.plot(self.t.numpy(), pred_y[0].numpy()[:, 0, 0], '-.r', label = 'N (Learned)')
            ax_x1.plot(self.t.numpy(), pred_y[0].numpy()[:, 0, 1], '-.g', label = 'P (Learned)')
            ax_x1.plot(self.t.numpy(), pred_y[0].numpy()[:, 0, 2], '-.b', label = 'Z (Learned)')

        ax_x1.set_xlim(self.t[0], self.t[-1])
        ax_x1.legend(bbox_to_anchor=(1.04,1), loc="upper left")

        plt.show() 

        if epoch != 0: 
            fig.savefig(os.path.join(self.figsave_dir, 'img'+str(epoch)))
            
### Initial Conditions
class initial_cond:

    def __init__(self, app):
        self.app = app

    def __call__(self, t):

        if self.app.bio_model == 'NPZ':
            x0 = [self.app.T_bio - 0.5*2, 0.5, 0.5]
        elif self.app.bio_model == 'NPZD':
            x0 = [self.app.T_bio - 0.5*2, 0.5, 0.5, 0.]
        elif self.app.bio_model == 'NNPZD':
            x0 = [self.app.T_bio/2., self.app.T_bio/2. - 2*0.5, 0.5, 0.5, 0.]
        return tf.convert_to_tensor([x0], dtype=tf.float32)