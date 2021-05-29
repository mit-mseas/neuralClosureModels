from src.solvers.neuralDDE_with_adjoint_accel import create_interpolator, create_validation_set

### Define a custom loss function
class custom_loss(tf.keras.losses.Loss):

    def call(self, true_y, pred_y):
        loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.math.squared_difference(pred_y, true_y), axis=-1)), axis=0)
        return loss

### Solve for the high resolution model
x_high_res = tf.linspace(0., args.L, args.nx_high_res)
t = tf.linspace(0., args.T, args.nt) # Time array

u0 = initial_cond(x_high_res, args.args_for_high_res) # Initial conditions

op = adeq.operators(args.args_for_high_res)

u_high_res = ddeinttf(adeq.ad_eqn(op, args), u0, t, alg_name = args.ode_alg_name, nsteps = args.nsteps)

# Compute FOM for the validation time
dt = t[1] - t[0]
val_t_len =  args.val_percentage * (t[-1] - t[0])
n_val = np.ceil(np.abs(val_t_len/dt.numpy())).astype(int)
val_t = tf.linspace(t[-1], t[-1] + val_t_len, n_val)

val_u_high_res = ddeinttf(adeq.ad_eqn(op, args), create_interpolator(u_high_res, t), val_t, alg_name = args.ode_alg_name, nsteps = args.nsteps)

print('High resolution model done!')

### Solve for low resolution model
x_low_res = tf.linspace(0., args.L, args.nx_low_res)

u0 = initial_cond(x_low_res, args.args_for_low_res) # Initial conditions

op = adeq.operators(args.args_for_low_res)

u_low_res = ddeinttf(adeq.ad_eqn(op, args), u0, t, alg_name = args.ode_alg_name, nsteps = args.nsteps)

val_u_low_res = ddeinttf(adeq.ad_eqn(op, args), create_interpolator(u_low_res, t), val_t, alg_name = args.ode_alg_name, nsteps = args.nsteps)

print('Low resolution model done!')

# Interpolate high resolution solution on low resolution grid

true_u_low_res = interp_high_res_to_low_res(u_high_res, x_high_res, x_low_res, t)

val_true_u_low_res = interp_high_res_to_low_res(val_u_high_res, x_high_res, x_low_res, val_t)

#### Change back to base directory
os.chdir(basedir)