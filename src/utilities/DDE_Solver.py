import numpy as np
import tensorflow as tf
import scipy.integrate
import scipy.interpolate
from IPython.core.debugger import set_trace

tf.keras.backend.set_floatx('float32')

class ddeVar:
    """
    The instances of this class are special function-like
    variables which store their past values in an interpolator and
    can be called for any past time: Y(t), Y(t-d).
    Very convenient for the integration of DDEs.
    """

    def __init__(self, g, tc=0., sign_flag=True): 
        """ g(t) = expression of Y(t) for t<tc """

        self.g = g
        self.tc = tc
        self.sign_flag = False if sign_flag < 0 else True
        self.data_shape = self.g(tc).numpy().shape
        # We must fill the interpolator with 2 points minimum
        self.interpolator = scipy.interpolate.interp1d(
            np.array([tc - 1, tc]),  # X 
            np.vstack([self.g(tc).numpy().flatten(), self.g(tc).numpy().flatten()]),  # Y 
            kind="linear",
            bounds_error=False, axis=0,
            fill_value=self.g(tc).numpy().flatten() 
        )

    def update(self, t, Y):
        """ Add one new (ti,yi) to the interpolator """
        
        idx = np.argwhere(self.interpolator.x == t)
        self.interpolator.x = np.delete(self.interpolator.x, idx)
        self.interpolator.y = np.delete(self.interpolator.y, idx, axis=0)

        if self.sign_flag: 
            t_stack = np.hstack([self.interpolator.x, [t]])
            Y_stack = np.vstack([self.interpolator.y, Y.flatten()])
        else :
            t_stack = np.hstack([[t], self.interpolator.x])
            Y_stack = np.vstack([Y.flatten(), self.interpolator.y])
            
        self.interpolator = scipy.interpolate.interp1d(
            t_stack,  # X
            Y_stack,  # Y
            kind="linear", axis=0,
            bounds_error=False,
            fill_value=Y.flatten(), assume_sorted = True
        )

    def __call__(self, t=0.):
        """ Y(t) will return the instance's value at time t """

        return self.g(t) if (t <= self.tc and self.sign_flag) or (t >= self.tc and not self.sign_flag) else tf.convert_to_tensor(np.reshape(self.interpolator(t), self.data_shape), tf.float32)
        
class dde(scipy.integrate.ode):
    """
    This class overwrites a few functions of ``scipy.integrate.ode``
    to allow for updates of the pseudo-variable Y between each
    integration step.
    """

    def __init__(self, f, jac=None):
        def f2(t, y, args):
            self.Y.update(t, y)
            return tf.reshape(f(self.Y, t, *args), [-1]).numpy()

        scipy.integrate.ode.__init__(self, f2, jac)
        self.set_f_params(None)

    def integrate(self, t, step=0, relax=0):
        scipy.integrate.ode.integrate(self, t, step, relax)
        self.Y.update(self.t, self.y)
        return np.reshape(self.y, self.Y.data_shape)

    def set_initial_value(self, Y):
        self.Y = Y  #!!! Y will be modified during integration
        scipy.integrate.ode.set_initial_value(self, Y(Y.tc).numpy().flatten(), Y.tc)
        
def ddeinttf(func, g, tt, fargs=None, alg_name = 'dopri5', nsteps = -1):
    """ Solves Delay Differential Equations

    Similar to scipy.integrate.odeint. Solves a Delay differential
    Equation system (DDE) defined by

        Y(t) = g(t) for t<0
        Y'(t) = func(Y,t) for t>= 0

    Where func can involve past values of Y, like Y(t-d).
    
    Adapted from: https://pypi.org/project/ddeint/
    Written by: Abhinav Gupta, MIT

    Parameters
    -----------
    
    func
      a function Y,t,args -> Y'(t), where args is optional.
      The variable Y is an instance of class ddeVar, which means that
      it is called like a function: Y(t), Y(t-d), etc. Y(t) returns
      a tensor.

    g
      The 'history function'. A function g(t)=Y(t) for t<0, g(t)
      returns a tensor.
    
    tt
      The tensor array of times [t0, t1, ...] at which the system must
      be solved.

    fargs
      Additional arguments to be passed to parameter ``func``, if any.


    Examples
    ---------
    
    We will solve the delayed Lotka-Volterra system defined as
    
        For t < 0:
        x(t) = 1+t
        y(t) = 2-t
    
        For t >= 0:
        dx/dt =  0.5* ( 1- y(t-d) )
        dy/dt = -0.5* ( 1- x(t-d) )
    
    The delay ``d`` is a tunable parameter of the model.
 
    >>> class model(tf.keras.Model):
    >>> def call(self, Y, t, d):
    >>>      x, y = Y(t)
    >>>      xd, yd = Y(t - d)
    >>>      return tf.expand_dims(tf.convert_to_tensor([0.5 * x * (1 - yd), -0.5 * y * (1 - xd)]), axis=0)
    >>> 
    >>> class values_before_zero(tf.keras.Model): # 'history' at t<0
    >>>  def call(self, t):
    >>>      return tf.expand_dims(tf.convert_to_tensor([1., 2.]), axis=0)
    >>> 
    >>> tt = tf.linspace(0.,30.,20000) # times for integration
    >>> d = tf.convet_to_tensor([0.5]) # set parameter d 
    >>> ic = values_before_zero()
    >>> func = model()
    >>> yy = ddeint(func,ic,tt,fargs=(d,)) # solve the DDE !
     
    """
    tt = tt.numpy()
    dde_ = dde(func)
    dde_.set_initial_value(ddeVar(g, tt[0], tf.sign(tt[-1] - tt[0]).numpy()))
    dde_.set_f_params(fargs if fargs else [])
    if nsteps < 0:
        dde_.set_integrator(alg_name)
    else:
        dde_.set_integrator(alg_name, nsteps = nsteps)
    results = [dde_.integrate(dde_.t + dt) for dt in np.diff(tt)]
    return tf.concat([tf.expand_dims(g(tt[0]), 0), tf.convert_to_tensor(results, tf.float32)], 0)        
