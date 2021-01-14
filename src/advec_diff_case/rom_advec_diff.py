
from src.utilities.DDE_Solver import ddeinttf 

from src.advec_diff_case.advec_diff_eqn import *

from IPython.core.debugger import set_trace

import numpy as np
import tensorflow as tf

tf.keras.backend.set_floatx('float32')


##### Class for user-defined parameters ####
class rom_eqn_args(ad_eqn_args):

    def __init__(self, T, nt, L, nx, Re, u_bc_0, u_bc_L, rom_dims, rom_batch_size, ad_eq_batch_size = 1):
        ad_eqn_args.__init__(self, T, nt, L, nx, Re, u_bc_0, u_bc_L, ad_eq_batch_size)

        self.rom_dims = rom_dims

        self.rom_batch_size = rom_batch_size

#### Create mean and modes
class create_mean_modes:
    def __init__(self, analy_sol, app, t):
        self.analy_sol = analy_sol
        self.app = app
        self.t = t

    def __call__(self):
        u_analy = []

        for i in range(self.app.nt):
            u_analy.append(self.analy_sol(self.t[i]))

        u_analy = tf.transpose(tf.concat(u_analy, axis=0))

        u_mean = tf.expand_dims(tf.reduce_mean(u_analy, axis=-1), axis=1)

        S, U, V = tf.linalg.svd(u_analy - tf.tile(u_mean, [1, self.app.nt]))

        ui = U[:, 0:self.app.rom_dims]

        return u_mean, ui


##### Initialozation of coefficients
class initial_cond_rom:

    def __init__(self, u0, ui, um):
        self.u0 = u0
        self.ui = ui
        self.um = tf.transpose(um, perm=[1, 0])

    def __call__(self, t):
        u0 = self.u0(t) - self.um
        ai_t0 = tf.cast(tf.einsum('ab, ca -> cb', tf.cast(self.ui, tf.float64), tf.cast(u0, tf.float64)), tf.float32)
        return ai_t0


##### R.H.S of da_dt
class rom_ad_eqn:

    def __init__(self, um, ui, op, app):
        self.um = tf.transpose(um, perm=[1, 0])
        self.ui = ui
        self.op = op
        self.app = app

        adv_mat, adv_bc = self.op.adv_uw(self.um)
        advec_um_um = tf.cast(tf.einsum('abc, ac -> ab', tf.cast(adv_mat, tf.float64), tf.cast(self.um, tf.float64)), tf.float32) + adv_bc

        self.ip_advec_um_um_ui = tf.cast(tf.einsum('ab, bc -> ac', tf.cast(advec_um_um, tf.float64), tf.cast(self.ui, tf.float64)), tf.float32)

        advec_um_ui = []
        for i in range(self.app.rom_dims):
            advec_um_ui.append(tf.einsum('abc, ac -> ab', tf.cast(adv_mat, tf.float64), tf.cast(tf.expand_dims(self.ui[:, i], axis=0), tf.float64)))

        advec_um_ui = tf.transpose(tf.concat(advec_um_ui, axis=0), perm=[1, 0])

        self.ip_advec_um_ui_ui = tf.einsum('ab, ac -> bc', advec_um_ui, tf.cast(self.ui, tf.float64))

        advec_ui_um = []
        for i in range(self.app.rom_dims):
            adv_mat, adv_bc = self.op.adv_cd(tf.expand_dims(self.ui[:, i], axis=0))
            advec_ui_um.append(tf.einsum('abc, ac -> ab', tf.cast(adv_mat, tf.float64), tf.cast(self.um, tf.float64)))

        advec_ui_um = tf.transpose(tf.concat(advec_ui_um, axis=0), perm=[1, 0])

        self.ip_advec_ui_um_ui = tf.einsum('ab, ac -> bc', advec_ui_um, tf.cast(self.ui, tf.float64))

        advec_ui_ui = []
        for i in range(self.app.rom_dims):
            for j in range(self.app.rom_dims):
                adv_mat, adv_bc = self.op.adv_cd(tf.expand_dims(self.ui[:, i], axis=0))
                advec_ui_ui.append(tf.einsum('abc, ac -> ab', tf.cast(adv_mat, tf.float64), tf.cast(tf.expand_dims(self.ui[:, j], axis=0), tf.float64)))

        advec_ui_ui = tf.transpose(tf.concat(advec_ui_ui, axis=0), perm=[1, 0])

        self.ip_advec_ui_ui_ui = tf.einsum('ab, ac -> bc', advec_ui_ui, tf.cast(self.ui, tf.float64))

        diff_um = tf.cast(tf.einsum('bc, ac -> ab', tf.cast(self.op.diff_mat, tf.float64), tf.cast(self.um, tf.float64)), tf.float32) + self.op.diff_bc

        self.ip_diff_um_ui = tf.einsum('ab, bc -> ac', tf.cast(diff_um, tf.float64), tf.cast(self.ui, tf.float64))

        diff_ui = []
        for i in range(self.app.rom_dims):
            diff_ui.append(tf.einsum('bc, ac -> ab', tf.cast(self.op.diff_mat, tf.float64), tf.cast(tf.expand_dims(self.ui[:, i], axis=0), tf.float64)))

        diff_ui = tf.transpose(tf.concat(diff_ui, axis=0), perm=[1, 0])

        self.ip_diff_ui_ui = tf.einsum('ab, ac -> bc', diff_ui, tf.cast(self.ui, tf.float64))

    def __call__(self, ai, t):

        ai_t = ai(t)

        ai_ai = tf.reshape(tf.einsum('ab, ac -> abc', tf.cast(ai_t, tf.float64), tf.cast(ai_t, tf.float64)), shape=[ai_t.shape[0], -1])

        daidt = - tf.tile(self.ip_advec_um_um_ui, [ai_t.shape[0], 1]) - tf.cast(tf.einsum('ab, bc -> ac', tf.cast(ai_t, tf.float64), self.ip_advec_ui_um_ui), tf.float32) \
                    - tf.cast(tf.einsum('ab, bc -> ac', tf.cast(ai_t, tf.float64), self.ip_advec_um_ui_ui), tf.float32) \
                    - tf.cast(tf.einsum('ab, bc -> ac', ai_ai, self.ip_advec_ui_ui_ui), tf.float32) + tf.cast(tf.tile(self.ip_diff_um_ui, [ai_t.shape[0], 1]), tf.float32) \
                    + tf.cast(tf.einsum('ab, bc -> ac', tf.cast(ai_t, tf.float64), self.ip_diff_ui_ui), tf.float32)

        return daidt