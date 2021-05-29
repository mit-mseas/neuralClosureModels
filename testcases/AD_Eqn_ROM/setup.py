from src.solvers.neuralDDE_with_adjoint_accel import create_interpolator, create_validation_set

### Define a custom loss function
class custom_loss(tf.keras.losses.Loss):

    def call(self, true_y, pred_y):
        loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.math.squared_difference(pred_y, true_y), axis=-1)), axis=0)
        return loss

### Solve for the full-order-model
x = tf.linspace(0., args.L, args.nx)
t = tf.linspace(0., args.T, args.nt) # Time array

u0 = initial_cond(x, args) # Initial conditions

op = adeq.operators(args)

u_fom = ddeinttf(adeq.ad_eqn(op, args), u0, t, alg_name = args.ode_alg_name, nsteps = args.nsteps)

# # Compute FOM for the validation time just to compute modes
# dt = t[1] - t[0]
# val_t_len_modes =  1. * (t[-1] - t[0])
# n_val_modes = np.ceil(np.abs(val_t_len_modes/dt.numpy())).astype(int)
# val_t_modes = tf.linspace(t[-1], t[-1] + val_t_len_modes, n_val_modes)

# val_u_fom_modes = ddeinttf(adeq.ad_eqn(op, args), create_interpolator(u_fom, t), val_t_modes, alg_name = args.ode_alg_name, nsteps = args.nsteps)

# Compute FOM for the validation time
dt = t[1] - t[0]
val_t_len =  args.val_percentage * (t[-1] - t[0])
n_val = np.ceil(np.abs(val_t_len/dt.numpy())).astype(int)
val_t = tf.linspace(t[-1], t[-1] + val_t_len, n_val)

val_u_fom = ddeinttf(adeq.ad_eqn(op, args), create_interpolator(u_fom, t), val_t, alg_name = args.ode_alg_name, nsteps = args.nsteps)

print('FOM done!')

### Project FOM on the modes
# Create modes for the training and validation period combined
u_mean, ui = create_mean_modes(tf.squeeze(tf.concat([u_fom, val_u_fom], axis=0), axis=1), args, tf.concat([t, val_t], axis=0))()
# u_mean, ui = create_mean_modes(tf.squeeze(tf.concat([u_fom, val_u_fom_modes], axis=0), axis=1), args, tf.concat([t, val_t_modes], axis=0))()

ai_t0 = rom.initial_cond_rom(u0, ui, u_mean)

true_ai = u_fom - tf.tile(tf.expand_dims(tf.transpose(u_mean, perm=[1, 0]), axis=0), [args.nt, args.multi_solve_size, 1])
true_ai = tf.cast(tf.einsum('ab, cda -> cdb', tf.cast(ui, tf.float64), tf.cast(true_ai, tf.float64)), tf.float32)

#Solve the ROM model
true_rom_model = rom.rom_ad_eqn(um = u_mean, ui = ui, op = op, app = args.rom_args_for_plot)
ai_whole = ddeinttf(true_rom_model, ai_t0, tf.concat([t, val_t], axis=0), alg_name = args.ode_alg_name, nsteps = args.nsteps)

#### Create validation set
val_obj = create_validation_set(ai_t0, t, args)

ai, val_ai = val_obj.data_split(ai_whole)

val_true_ai = val_u_fom - tf.tile(tf.expand_dims(tf.transpose(u_mean, perm=[1, 0]), axis=0), [val_obj.val_t.shape[0], args.multi_solve_size, 1])
val_true_ai = tf.cast(tf.einsum('ab, cda -> cdb', tf.cast(ui, tf.float64), tf.cast(val_true_ai, tf.float64)), tf.float32)

u0_red = red_initial_cond(ai_t0, u_mean, ui)

u_fom_red_ic = ddeinttf(adeq.ad_eqn(op, args), u0_red, tf.concat([t, val_t], axis=0), alg_name = args.ode_alg_name, nsteps = args.nsteps)

true_ai_red = u_fom_red_ic - tf.tile(tf.expand_dims(tf.transpose(u_mean, perm=[1, 0]), axis=0), [args.nt + n_val, args.multi_solve_size, 1])
true_ai_red = tf.cast(tf.einsum('ab, cda -> cdb', tf.cast(ui, tf.float64), tf.cast(true_ai_red, tf.float64)), tf.float32)

true_ai_red, val_true_ai_red = val_obj.data_split(true_ai_red)

#### Change back to base directory
os.chdir(basedir)