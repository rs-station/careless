import numpy as np
import math
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import special_math
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
from tensorflow.python.ops import array_ops
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import special_math
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.bijectors import exp as exp_bijector
from tensorflow_probability.python.bijectors import absolute_value as abs_bijector
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import samplers
from tensorflow_probability import math as tfm
from tensorflow_probability.python.internal import tensor_util



class Rice(tfd.Distribution):
    #Value of nu/sigma for which, above with the pdf and moments will be swapped with a normal distribution
    _normal_crossover = 10. 
    def __init__(self,
		   nu,
		   sigma,
		   validate_args=False,
		   allow_nan_stats=True,
		   name='Rice'):
      parameters = dict(locals())
      with tf.name_scope(name) as name:
          dtype = dtype_util.common_dtype([nu, sigma], dtype_hint=tf.float32)
          self._nu = tensor_util.convert_nonref_to_tensor(
              nu, dtype=dtype, name='nu')
          self._sigma = tensor_util.convert_nonref_to_tensor(
              sigma, dtype=dtype, name='sigma')
          super(Rice, self).__init__(
            dtype=dtype,
            reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            name=name)

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        # pylint: disable=g-long-lambda
        return dict(
            nu=parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=(
                    lambda: exp_bijector.Exp())),
            sigma=parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=(
                    lambda: exp_bijector.Exp())),
            )
        # pylint: enable=g-long-lambda

    def moment(self, t):
        from scipy.stats import rice,norm
        nu,sigma = self.nu,self.sigma
        snr = nu / sigma
        safe_sigma = np.minimum(sigma, self._normal_crossover * nu)
        idx = snr <= self._normal_crossover
        safe_sigma = np.where(
            idx,
            sigma,
            nu / self._normal_crossover,
        )
        out = np.where(
            idx,
            rice.moment(t, nu/safe_sigma, scale=safe_sigma), #use safe sigma to prevent warnings
            norm.moment(t, loc=nu, scale=nu),
        )
        return out

    @property
    def nu(self):
        return self._nu

    @property
    def sigma(self):
        return self._sigma

    def _batch_shape_tensor(self, nu, sigma):
        return array_ops.shape(nu / sigma)

    def _sample_n(self, n, seed=None):
        seed = samplers.sanitize_seed(seed)
        nu = tf.convert_to_tensor(self.nu)
        sigma = tf.convert_to_tensor(self.sigma)
        shape = ps.concat([[n], self._batch_shape_tensor(nu=nu, sigma=sigma)], axis=0)
        return stateless_rice(shape, nu, sigma, seed)

    @staticmethod
    def _log_bessel_i0(x):
        return tf.math.log(tf.math.bessel_i0e(x)) + tf.math.abs(x)

    @staticmethod
    def _log_bessel_i1(x):
        return tf.math.log(tf.math.bessel_i1e(x)) + tf.math.abs(x)

    @staticmethod
    def _bessel_i0(x):
        return tf.math.exp(Rice._log_bessel_i0(x))

    @staticmethod
    def _bessel_i1(x):
        return tf.math.exp(Rice._log_bessel_i1(x))

    @staticmethod
    def _laguerre_half(x):
        return (1. - x) * tf.math.exp(x / 2. + Rice._log_bessel_i0(-0.5 * x)) - x * tf.math.exp(x / 2.  + Rice._log_bessel_i1(-0.5 * x) )

    def prob(self, X):
        return tf.math.exp(self.log_prob(X))

    def log_prob(self, X):
        sigma = self.sigma
        nu = self.nu
        log_p = tf.math.log(X) - 2.*tf.math.log(sigma) - (X**2. + nu**2.)/(2*sigma**2.) + \
                    self._log_bessel_i0(X * nu/sigma**2.)
        return log_p

    def mean(self):
        sigma = self.sigma
        nu = self.nu
        mean = sigma * tf.math.sqrt(np.pi / 2.) * self._laguerre_half(-0.5*(nu/sigma)**2)
        return tf.where(nu/sigma > self._normal_crossover,  nu, mean)

    def variance(self):
        sigma = self.sigma
        nu = self.nu
        variance = 2*sigma**2. + nu**2. - 0.5*np.pi * sigma**2. * self._laguerre_half(-0.5*(nu/sigma)**2)**2.
        return tf.where(nu/sigma > self._normal_crossover,  sigma**2., variance)

    def stddev(self):
        return tf.math.sqrt(self.variance())

    @staticmethod
    def sample_gradients(z, nu, sigma):
        log_z, log_nu, log_sigma = tf.math.log(z), tf.math.log(nu), tf.math.log(sigma)
        log_a = log_nu - log_sigma
        log_b = log_z - log_sigma
        ab = tf.exp(log_a + log_b)  # <-- argument of bessel functions
        log_i0 = Rice._log_bessel_i0(ab)
        log_i1 = Rice._log_bessel_i1(ab)

        dnu = tf.exp(log_i1 - log_i0)
        dsigma = -(
            tf.exp(log_nu - log_sigma + log_i1 - log_i0)
            - tf.exp(log_z - log_sigma)
        )
        return dnu, dsigma


@tf.custom_gradient
def stateless_rice(shape, nu, sigma, seed):
    A = tf.random.stateless_normal(shape, seed, mean=nu, stddev=sigma)
    B = tf.random.stateless_normal(shape, seed, mean=tf.zeros_like(nu), stddev=sigma)
    z = tf.sqrt(A*A + B*B)
    def grad(upstream):
        dnu,dsigma = Rice.sample_gradients(z, nu, sigma)
        dnu = tf.reduce_sum(upstream * dnu, axis=0)
        dsigma = tf.reduce_sum(upstream * dsigma, axis=0)
        return None, dnu, dsigma, None
    return z, grad



def folded_normal_sample_gradients(z, loc, scale):
    alpha = (z + loc) / scale
    beta = (z - loc) / scale

    p_a = tfd.Normal(loc, scale).prob(z)   #N(z|loc,scale)
    p_b = tfd.Normal(-loc, scale).prob(z)  #N(z|-loc,scale)

    # This formula is dz = N(z|-loc,scale) + N(z|loc,scale)
    dz = p_a + p_b

    # This formula is dloc = N(z|-loc,scale) - N(z|loc,scale)
    dloc = p_b - p_a
    dloc = dloc / dz

    # This formula is -[b * N(z|-loc, scale) + a * N(z|loc, scale)]
    dscale = -alpha * p_b - beta * p_a
    dscale = dscale / dz
    return dloc, dscale

@tf.custom_gradient
def stateless_folded_normal(shape, loc, scale, seed):
    z = tf.random.stateless_normal(shape, seed, mean=loc, stddev=scale)
    z = tf.abs(z)
    def grad(upstream):
        grads = folded_normal_sample_gradients(z, loc, scale)
        dloc,dscale = grads[0], grads[1]
        dloc = tf.reduce_sum(-upstream * dloc, axis=0)
        dscale = tf.reduce_sum(-upstream * dscale, axis=0)
        return None, dloc, dscale, None
    return z, grad

class FoldedNormal(tfd.Distribution):
    """The folded normal distribution."""
    _normal_crossover = 10.
    def __init__(self,
               loc,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name='FoldedNormal'):
        """Construct a folded normal distribution.
        Args:
          loc: Floating-point `Tensor`; the means of the underlying
            Normal distribution(s).
          scale: Floating-point `Tensor`; the stddevs of the underlying
            Normal distribution(s).
          validate_args: Python `bool`, default `False`. Whether to validate input
            with asserts. If `validate_args` is `False`, and the inputs are
            invalid, correct behavior is not guaranteed.
          allow_nan_stats: Python `bool`, default `True`. If `False`, raise an
            exception if a statistic (e.g. mean/mode/etc...) is undefined for any
            batch member If `True`, batch members with valid parameters leading to
            undefined statistics will return NaN for this statistic.
          name: The name to give Ops created by the initializer.
        """
        parameters = dict(locals())
        with tf.name_scope(name) as name:
          dtype = dtype_util.common_dtype([loc, scale], dtype_hint=tf.float32)
          self._loc = tensor_util.convert_nonref_to_tensor(
              loc, dtype=dtype, name='loc')
          self._scale = tensor_util.convert_nonref_to_tensor(
              scale, dtype=dtype, name='scale')
          super(FoldedNormal, self).__init__(
              dtype=dtype,
              reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
              validate_args=validate_args,
              allow_nan_stats=allow_nan_stats,
              parameters=parameters,
              name=name)

    def _batch_shape_tensor(self, loc, scale):
        return array_ops.shape(loc / scale)

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        # pylint: disable=g-long-lambda
        return dict(
            loc=parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=(
                    lambda: exp_bijector.Exp())),
            scale=parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=(
                    lambda: exp_bijector.Exp())),
            )
        # pylint: enable=g-long-lambda

    @property
    def loc(self):
        """Distribution parameter for the pre-transformed mean."""
        return self._loc

    @property
    def scale(self):
        """Distribution parameter for the pre-transformed standard deviation."""
        return self._scale

    @property
    def _pi(self):
        return tf.convert_to_tensor(np.pi, self.dtype)

    def _cdf(self, x):
        loc,scale = self.loc,self.scale
        x = tf.convert_to_tensor(x)
        a = (x + loc) / scale
        b = (x - loc) / scale
        ir2 = tf.constant(tf.math.reciprocal(tf.sqrt(2.)), dtype=x.dtype)
        return 0.5 * (tf.math.erf(ir2 * a) - tf.math.erf(ir2 * b))

    def _sample_n(self, n, seed=None):
        seed = samplers.sanitize_seed(seed)
        loc = tf.convert_to_tensor(self.loc)
        scale = tf.convert_to_tensor(self.scale)
        shape = ps.concat([[n], self._batch_shape_tensor(loc=loc, scale=scale)], axis=0)
        return stateless_folded_normal(shape, loc, scale, seed)

    def _log_prob(self,  value):
        loc,scale = self.loc,self.scale
        result = tfm.log_add_exp(
            tfd.Normal(loc, scale).log_prob(value), 
            tfd.Normal(-loc, scale).log_prob(value),
        )
        return result
        #return tf.where(value < 0, tf.constant(-np.inf, dtype=result.dtype), result)

    @staticmethod
    def _folded_normal_mean(loc, scale):
        u,s = loc, scale
        c = loc / scale
        return s * tf.sqrt(2/math.pi) * tf.math.exp(-0.5 * c * c) + u * tf.math.erf(c/math.sqrt(2))

    def _mean(self):
        u = self.loc
        s = self.scale
        c = u/s

        idx = tf.abs(c) >= 10.
        s_safe = tf.where(idx, 1., s)
        u_safe = tf.where(idx, 1., u)
        c_safe = tf.where(idx, 1., c)

        return tf.where(
            c >= self._normal_crossover,
            u,
            self._folded_normal_mean(u_safe, s_safe)
        )

    def _variance(self):
        u = self.loc
        s = self.scale
        c = u/s

        idx = tf.abs(c) >= 10.
        s_safe = tf.where(idx, 1., s)
        u_safe = tf.where(idx, 1., u)
        c_safe = tf.where(idx, 1., c)
        m = self._folded_normal_mean(u_safe, s_safe)

        return tf.where(
            idx, 
            s*s,
            u*u + s*s - m*m,
        )

    def moment(self, t):
        """ Use Scipy to calculate the t-moment of a folded normal """
        from scipy.stats import foldnorm,norm
        loc,scale = self.loc.numpy(),self.scale.numpy()
        c = loc / scale
        idx = np.abs(c) > self._normal_crossover
        c_safe = tf.where(idx, c, 0.)
        scale_safe = tf.where(idx, scale, 1.)

        result = np.where(
            idx,
            foldnorm.moment(t, loc/scale, scale=scale),
            norm.moment(t, loc=loc, scale=scale),
        )
        return result
