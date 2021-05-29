from src.solvers.neuralDDE_with_adjoint_accel import create_interpolator, create_validation_set

### Define a custom loss function
class custom_loss:
    
    def __init__(self, args):
        self.args = args

    def __call__(self, true_y, pred_y):
        
        zero_places = tf.logical_or(tf.less(pred_y, tf.constant([0.])), tf.greater(pred_y, tf.constant([self.args.T_bio_max])))
        mask_tensor = tf.where(zero_places, 1., 0.)
        
        loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.math.squared_difference(pred_y, true_y), axis=-1)), axis=0) \
                + 1. * tf.reduce_mean(tf.reduce_sum(mask_tensor, axis=-1), axis=0)
        return loss

### Solve for the high complexity model
# t_M = np.array([0, 75, 100, 250, 365, 365 + 75, 365 + 100, 365 + 250, 365 + 365])
# M = np.array([-60., -80., -25., -25., -60., -80., -25., -25., -60.])

# M = np.array([-60., -60., -60., -60., -60., -60., -60., -60., -60.])

# t_M = np.array([0, 75-25, 100-25, 250-25, 365, 365 + 75-25, 365 + 100-25, 365 + 250-25, 365+365])
# M = np.array([-70., -80., -25., -25., -70, -80., -25., -25., -70])

t_M = np.array([0, 25, 175, 365, 365 + 25, 365 + 175, 365+365])
M = np.array([-80., -25., -25., -80., -25., -25., -80])

plt.plot(t_M, M)
plt.show

K_z_obj = diff_coeff(args, M, t_M, 'linear')

t = tf.linspace(0., args.T, args.nt) # Time array

x0_high_complex = initial_cond(args.bio_args_for_high_complex) # Initial conditions

x_high_complex = ddeinttf(bio.bio_eqn(args.bio_args_for_high_complex, K_z_obj), x0_high_complex, t, alg_name = args.ode_alg_name, nsteps = args.nsteps)



# Compute FOM for the validation time
dt = t[1] - t[0]
val_t_len =  args.val_percentage * (t[-1] - t[0])
n_val = np.ceil(np.abs(val_t_len/dt.numpy())).astype(int)
val_t = tf.linspace(t[-1], t[-1] + val_t_len, n_val)

val_x_high_complex = ddeinttf(bio.bio_eqn(args.bio_args_for_high_complex, K_z_obj), create_interpolator(x_high_complex, t), val_t, alg_name = args.ode_alg_name, nsteps = args.nsteps)

print('Higher complexity model done!')



### Transform states of high complexity model to low complexity model

# Create modes for the training and validation period combined
true_x_low_complex = bio.convert_high_complex_to_low_complex_states(x_high_complex, args)

x0_low_complex = initial_cond(args)


# Solve the low complexity model
x_low_complex = ddeinttf(bio.bio_eqn(args, K_z_obj), x0_low_complex, t, alg_name = args.ode_alg_name, nsteps = args.nsteps)

val_x_low_complex = ddeinttf(bio.bio_eqn(args, K_z_obj), create_interpolator(x_low_complex, t), val_t, alg_name = args.ode_alg_name, nsteps = args.nsteps)

#### Create validation set
val_obj = create_validation_set(x0_low_complex, t, args)

val_true_x_low_complex = bio.convert_high_complex_to_low_complex_states(val_x_high_complex, args)


#### Change back to base directory
os.chdir(basedir)