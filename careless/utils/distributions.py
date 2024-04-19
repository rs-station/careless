import tensorflow as tf
from tensorflow_probability.python.internal import special_math
from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp
from tensorflow_probability import bijectors as tfb
from tensorflow_probability.python.internal import tensor_util
import numpy as np


class Amoroso(tfd.TransformedDistribution):
    def __init__(self,
		   a,
		   theta,
		   alpha,
                   beta,
		   validate_args=False,
		   allow_nan_stats=True,
		   name='Amoroso'):

        parameters = dict(locals())
        with tf.name_scope(name) as name:
            self._a = tensor_util.convert_nonref_to_tensor(a)
            self._theta = tensor_util.convert_nonref_to_tensor(theta)
            self._alpha = tensor_util.convert_nonref_to_tensor(alpha)
            self._beta = tensor_util.convert_nonref_to_tensor(beta)
            gamma = tfd.Gamma(alpha, 1.)

            chain = tfb.Invert(tfb.Chain([
                tfb.Exp(),
                tfb.Scale(beta),
                tfb.Shift(-tf.math.log(theta)),
                tfb.Log(),
                tfb.Shift(-a),
            ]))

            super().__init__(
                    distribution=gamma, 
                    bijector=chain, 
                    validate_args=validate_args,
                    parameters=parameters,
                    name=name)

    @property
    def a(self):
        return self._a

    @property
    def theta(self):
        return self._theta

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    def custom_log_prob(self, X):
        a = self.a
        theta = self.theta
        alpha = self.alpha
        beta  = self.beta

        inflation = 1e10
        x = tf.math.log(tf.abs(beta)) - tf.math.log(tf.abs(theta)) - tf.math.lgamma(a) 
        y = (alpha*beta - 1.)*(tf.math.log(inflation*X - inflation*a) - tf.math.log(theta) - tf.math.log(inflation)) 
        z = - tf.math.pow(X - a, beta) * tf.math.exp(- beta * tf.math.log(theta))
        print(x,y,z)

        return x + y + z

    def mean(self):
        """
        The mean of of the Amoroso distribution exists for `alpha + 1/beta >= 0`.
        It can be computed analytically by

        ```
        mean = a + theta * gamma(alpha + 1/beta) / gamma(alpha)
        ```
        """
        a,theta,alpha,beta = self.a,self.theta,self.alpha,self.beta
        return a + tf.math.exp(tf.math.log(theta)+tf.math.lgamma(alpha + tf.math.reciprocal(beta)) - tf.math.lgamma(alpha))

    def variance(self):
        """
        The variance of of the Amoroso distribution exists for `alpha + 2/beta >= 0`.
        It can be computed analytically by

        ```
        variance = theta**2 * (gamma(alpha + 2/beta) / gamma(alpha) - gamma(alpha + 1/beta)**2 / gamma(alpha)**2 )
        ```
        """
        theta,alpha,beta = self.theta,self.alpha,self.beta
        return theta**2. * (tf.math.exp(tf.math.lgamma(alpha + 2./beta) - tf.math.lgamma(alpha)) - tf.math.exp(2.*tf.math.lgamma(alpha + 1/beta) - 2.*tf.math.lgamma(alpha)))

    def stddev(self):
        """
        The variance of of the Amoroso distribution exists for `alpha + 2/beta >= 0`.
        It can be computed analytically by

        ```
        variance = theta**2 * (gamma(alpha + 2/beta) / gamma(alpha) - gamma(alpha + 1/beta)**2 / gamma(alpha)**2 )
        ```

        The standard deviation is computed by
        ```amoroso.stddev() = tf.sqrt(amoroso.variance())```
        """
        return tf.math.sqrt(self.variance())


class Stacy(Amoroso):
    def __init__(self,
		 theta,
		 alpha,
                 beta,
		 validate_args=False,
		 allow_nan_stats=True,
		 name='Stacy'):

        parameters = dict(locals())
        with tf.name_scope(name) as name:
            super().__init__(
                0.,
                theta,
                alpha,
                beta,
                validate_args,
                allow_nan_stats,
                name,
            )

    @classmethod
    def wilson_prior(cls, centric, epsilon, Sigma=1.):
        """
        Construct a wilson prior based ont he Stacy distribution.
        Centric Wilson distributions are HalfNormals with scale = sqrt(epsilon * Sigma)
        P(F|Sigma,epsilon)  = (2*pi*Sigma*epsilon)**(-0.5) * exp(-F**2 / 2 / Sigma / epsilon)
                            = Stacy(F| sqrt(2*Sigma*epsilon), 0.5, 2)

        Acentric Wilson distribution 
        P(F|Sigma,epsilon) = (2/Sigma/epsilon) * F * exp(-(F**2/Sigma/epsilon))
                           = Stacy(F | sqrt(Sigma*epsilon), 1., 2.)

        Parameters
        ----------
        centric : array (bool)
            boolean array with True for centric reflections
        epsilon : array (float)
            float array of structure factor multiplicities
        """
        centric = tf.cast(centric, dtype=tf.float32)
        epsilon = tf.convert_to_tensor(epsilon, dtype=tf.float32)
        #centric = np.array(centric, dtype=np.float32) #<-- coerce same same
        theta = centric*np.sqrt(2. * epsilon * Sigma) + (1.-centric)*np.sqrt(Sigma*epsilon)
        alpha = centric*0.5 + (1.-centric)
        beta = centric*2. + (1. - centric)*2.
        return cls(theta, alpha, beta)

    @staticmethod
    def _stacy_params(dist):
        if isinstance(dist, Stacy):
            params = (dist.theta, dist.alpha, dist.beta)
        elif isinstance(dist, tfd.Weibull):
            # Weibul(x; k, lambda) = Stacy(x; lambda, 1, k)
            k = dist.concentration
            lam = dist.scale
            params = (lam, 1., k)
        elif isinstance(dist, tfd.HalfNormal):
            #HalfNormal(x; scale) = Stacy(x; sqrt(2)*scale, 0.5, 2)
            scale = dist.scale
            params = (np.sqrt(2.) * scale, 0.5, 2.)
        else:
            raise TypeError(f"Equivalent Stacy parameters cannot be determined for distribution, {dist}. " 
                             "Only tfd.Weibull, tfd.HalfNormal, or Stacy can be converted to Stacy parameterisation")
        return params

    @staticmethod
    def _bauckhage_params(dist):
        theta, alpha, beta = Stacy._stacy_params(dist)
        bauckhage_params = (theta, alpha*beta, beta)
        return bauckhage_params 

    def kl_divergence(self, other):
        """
        The Stacy distribution has an analytical KL div. 
        However, it isn't documented in the same parameterization as the Crooks Amoroso tome. 
        To avoid confusion, I will first translate the Stacy distributions from 
        Crooks's parameterization to the one in the KL div paper. 

        ```
        Stacy(x; a,d,p) = x**(d-1) * exp(-(x/a)**p) / Z
        ```
        where
        ```
        a = theta
        d = alpha * beta
        p = beta
        ```
        Then the KL div is
        ```
        log(p1*a2**d2*gamma(d2/p2)) - log(p2*a1**d1*gamma(d1/p1)) + 
        (digamma(d1/p1)/p1 + log(a1)) * (d1 - d2) + 
        gamma((d1 + p2)/p1) * (a1/a2)**p2 / gamma(d1/p1) - d1/p1
        ```

        See Bauckhage 2014 for derivation. 
        https://arxiv.org/pdf/1401.6853.pdf

        Parameters
        ----------
        other : Stacy or tfd.Weibull or tfd.HalfNormal
        """
        a1,d1,p1 = self._bauckhage_params(self)
        a2,d2,p2 = self._bauckhage_params(other)

        #The following numerics are easier to read if you alias this
        ln = tf.math.log

        kl = ln(p1) + d2*ln(a2) + tf.math.lgamma(d2/p2) - ln(p2) - d1*ln(a1) - tf.math.lgamma(d1/p1) + \
             (tf.math.digamma(d1/p1)/p1 + ln(a1))*(d1 - d2) +  \
             tf.math.exp(tf.math.lgamma((d1 + p2)/p1) - tf.math.lgamma(d1/p1) + p2*(ln(a1) - ln(a2))) \
             - d1/p1

        return kl


class Rice(tfd.Distribution):
    def __init__(self,
		   nu,
		   sigma,
		   validate_args=False,
		   allow_nan_stats=True,
		   name='Rice'):

        parameters = dict(locals())
        #Value of nu/sigma for which, above with the pdf and moments will be swapped with a normal distribution
        self._normal_crossover = 40. 
        with tf.name_scope(name) as name:
            self._nu = tensor_util.convert_nonref_to_tensor(nu)
            self._sigma = tensor_util.convert_nonref_to_tensor(sigma)
            self._base_normal = tfd.Normal(0., self._sigma)

    @property
    def nu(self):
        return self._nu

    @property
    def sigma(self):
        return self._sigma

    def sample(self, sample_shape=(), seed=None, name='sample', **kwargs):
        s1 = self._base_normal.sample(sample_shape=sample_shape, seed=seed, name=name, **kwargs)
        s2 = self._base_normal.sample(sample_shape=sample_shape, seed=seed, name=name, **kwargs)
        return tf.math.sqrt(
                    s1**2. +
                   (s2 + self.nu)**2.
               )

    def _log_bessel_i0(self, x):
        return tf.math.log(tf.math.bessel_i0e(x)) + tf.math.abs(x)

    def _log_bessel_i1(self, x):
        return tf.math.log(tf.math.bessel_i1e(x)) + tf.math.abs(x)

    def _bessel_i0(self, x):
        return tf.math.exp(self._log_bessel_i0(x))

    def _bessel_i1(self, x):
        return tf.math.exp(self._log_bessel_i1(x))

    def _laguerre_half(self, x):
        return (1. - x) * tf.math.exp(x / 2. + self._log_bessel_i0(-0.5 * x)) - x * tf.math.exp(x / 2.  + self._log_bessel_i1(-0.5 * x) )

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

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import special_math
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import prefer_static as ps
import numpy as np
from tensorflow_probability.python.bijectors import exp as exp_bijector
from tensorflow_probability.python.bijectors import absolute_value as abs_bijector
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import samplers
from tensorflow_probability import math as tfm
from tensorflow_probability.python.internal import tensor_util

def _logspace_sample_gradients(z, loc, scale):
    alpha_sign,log_alpha = tf.sign(z + loc), tf.math.log(tf.abs(z + loc)) - tf.math.log(scale)
    beta_sign,log_beta = tf.sign(z - loc), tf.math.log(tf.abs(z - loc)) - tf.math.log(scale)

    log_p_a = tfd.Normal(loc, scale).log_prob(z)   #N(z|loc,scale)
    log_p_b = tfd.Normal(-loc, scale).log_prob(z)  #N(z|-loc,scale)

    # This formula is dz = N(z|-loc,scale) + N(z|loc,scale)
    log_dz = tfp.math.log_add_exp(log_p_a, log_p_b)

    # This formula is dloc = N(z|-loc,scale) - N(z|loc,scale)
    log_dloc,dloc_sign = tfp.math.log_sub_exp(log_p_b, log_p_a, return_sign=True)
    # dloc = dloc / dz
    dloc = dloc_sign * tf.exp(log_dloc - log_dz)

    # This formula is -[b * N(z|-loc, scale) + a * N(z|loc, scale)]
    dscale = -alpha_sign * tf.exp(log_alpha + log_p_b - log_dz) - beta_sign * tf.exp(log_beta + log_p_a - log_dz)
    return dloc, dscale
    #return log_dz, dloc, dscale

def _sample_gradients(z, loc, scale):
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


def sample_gradients(z, loc, scale, logspace=True):
    if logspace:
        return _logspace_sample_gradients(z, loc, scale)
    return _sample_gradients(z, loc, scale)

@tf.custom_gradient
def stateless_folded_normal(shape, loc, scale, seed):
    z = tf.random.stateless_normal(shape, seed, mean=loc, stddev=scale)
    z = tf.abs(z)
    def grad(upstream):
        grads = sample_gradients(z, loc, scale)
        dloc,dscale = grads[0], grads[1]
        dloc = tf.reduce_sum(-upstream * dloc, axis=0)
        dscale = tf.reduce_sum(-upstream * dscale, axis=0)
        return None, dloc, dscale, None
    return z, grad

class FoldedNormal(tfd.Distribution):
    """The folded normal distribution."""
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

    def _mean(self):
        u = self.loc
        s = self.scale
        snr = u/s
        return s * tf.sqrt(2/self._pi) * tf.math.exp(-0.5 * tf.square(snr)) + u * (1. - 2. * special_math.ndtr(-snr))

    def _variance(self):
        u = self.loc
        s = self.scale
        return tf.square(u) + tf.square(s) - tf.square(self.mean())

    def sample_square(self, sample_shape=(), seed=None, name='sample', **kwargs):
        z = self.distribution.sample(sample_shape, seed, name, **kwargs)
        return tf.square(z)

    def moment(self, t, npoints=100):
        """ use quadrature to estimate E[X^t] where X ~ FoldedNormal """
        loc, scale = self.loc,self.scale
        window_size = 20.0
        Jmin = loc - window_size * scale
        Jmax = loc + window_size * scale
        Jmin = tf.maximum(0., Jmin)

        grid, weights = np.polynomial.chebyshev.chebgauss(npoints)
        grid, weights = tf.convert_to_tensor(grid, loc.dtype),tf.convert_to_tensor(weights, loc.dtype)
        grid = tf.reshape(
            grid, 
            (npoints,) + tf.ones_like(tf.shape(Jmin)),
        )

        # Note -- grid dimensions need to be leading for log_prob to work properly
        logweights = (0.5 * (tf.math.log(1 - grid) + tf.math.log(1 + grid)) + tf.math.log(weights))[None, ...]
        J = (Jmax - Jmin)[..., None] * grid / 2.0 + (Jmax + Jmin)[..., None] / 2.0
        log_J = np.log(J)
        log_prefactor = np.log(Jmax - Jmin) - np.log(2.0)
        log_p = self.log_prob(J)
        t * log_J + log_p


        Phi = special_math.ndtr
        mom = (0.5 * scale * scale * t * t + loc * t) * tf.math.log(
            1. - Phi(-loc/scale - scale*t) + tf.exp(-2. * loc * t) * (1. - Phi(loc/scale - scale*t))
        )
        #mom = tf.exp(0.5 * scale * scale * t * t + loc * t) * special_math.ndtr(loc / scale + scale * t) + \
        #      tf.exp(0.5 * scale * scale * t * t - loc * t) * special_math.ndtr(-loc / scale + scale * t)
        return mom

