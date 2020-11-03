from careless.utils.distributions import Rice,FoldedNormal
from tensorflow_probability.python.internal.special_math import ndtr
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
from tensorflow_probability.python.internal import tensor_util
import tensorflow as tf
import numpy as np

class RiceWoolfson(tfd.Distribution):
    def __init__(self, loc, scale, centric):
        """
        This is a hybrid distribution to parameterize posteriors over structure factors. 
        It uses the Rice distribution to model acentric structure factors and
        the Folded normal or "Woolfson" distribution to model centrics. 

        Parameters
        ----------
        loc : array (float)
            location parameter for the distributions
        scale : array (float)
            scale parameter for the distributions
        centric : array (float;bool)
            Array that same length as loc and scale that is 1./True for centric reflections and 0./False
        """
        self._loc   = tensor_util.convert_nonref_to_tensor(loc, dtype=tf.float32)
        self._scale = tensor_util.convert_nonref_to_tensor(scale, dtype=tf.float32)
        self._centric = np.array(centric, dtype=np.bool)
        self._woolfson = FoldedNormal(self._loc, self._scale)
        self._rice = Rice(self._loc, self._scale)
        self.eps = np.finfo(np.float32).eps

    def mean(self):
        return tf.where(self._centric, self._woolfson.mean(), self._rice.mean())

    def variance(self):
        return tf.where(self._centric, self._woolfson.variance(), self._rice.variance())

    def stddev(self):
        return tf.where(self._centric, self._woolfson.stddev(), self._rice.stddev())

    def sample(self, sample_shape=(), seed=None, name='sample', **kwargs):
        return tf.where(self._centric, self._woolfson.sample(sample_shape, seed, name, **kwargs)+self.eps, self._rice.sample(sample_shape, seed, name, **kwargs))

    def log_prob(self, x):
        return tf.where(self._centric, self._woolfson.log_prob(x), self._rice.log_prob(x))

    def prob(self, x):
        return tf.where(self._centric, self._woolfson.prob(x), self._rice.prob(x))


#This is a temporary workaround for tfd.TruncatedNormal which has a bug in sampling
#2020-10-30: This should be removed if this issue is fixed: https://github.com/tensorflow/probability/issues/1149
#
#2020-11-01: On second thought, this may not be fixed unless they git rid of the current rejection sampler based 
# implementation. See https://github.com/tensorflow/probability/issues/518, for additional issues.

class TruncatedNormal(tfd.TruncatedNormal):
    def sample(self, *args, **kwargs):
        s = super().sample(*args, **kwargs)
        low = self.low
        return tf.maximum(self.low, s)

from tensorflow_probability.python.internal import tensor_util
class HybridTruncatedNormal(tfd.TruncatedNormal):
    def __init__(self, loc, scale, low, high, *args, **kwargs):
        super().__init__(loc, scale, low, high, *args, **kwargs)
        self._crossover = 10.
        self._normal = tfd.Normal(loc, scale)

    def log_prob(self, X, *args, **kwargs):
        loc,scale,low,high = self._loc_scale_low_high()
        alpha = (low - loc)  / scale
        beta  = (high - loc) / scale
        use_normal = (alpha < -self._crossover) & (beta > self._crossover)
        return tf.where(use_normal, self._normal.log_prob(X, *args, **kwargs), tf.where(use_normal, 0., super().log_prob(X, *args, **kwargs)))

    def sample(self, *args, **kwargs):
        loc,scale,low,high = self._loc_scale_low_high()

        normal_samples = self._normal.sample(*args, **kwargs)
        alpha = (low - loc)  / scale
        beta  = (high - loc) / scale

        uniform_samples = ndtr((normal_samples - loc)/scale)
        transformed_samples = tf.math.ndtri(ndtr(alpha) + uniform_samples * (ndtr(beta) - ndtr(alpha))) * scale + loc
        use_normal = (alpha < -self._crossover) & (beta > self._crossover)
        return tf.where(use_normal, normal_samples, tf.where(use_normal, 0., transformed_samples))


class ShiftedFoldedNormal(tfd.TransformedDistribution):
    def __init__(self, loc, scale, low, *args, **kwargs):
        self._low = tensor_util.convert_nonref_to_tensor(low)
        parent = FoldedNormal(loc, scale)
        bijector = tfb.Shift(self._low)
        super().__init__(
            parent,
            bijector,
            *args,
            **kwargs,
        )

    @property
    def low(self):
        return self._loc

    def stddev(self):
        return self.distribution.stddev()

#class TruncatedNormal(tfd.Uniform):
#    def __init__(self,
#		   loc,
#		   scale,
#                   low,
#                   high,
#		   validate_args=False,
#		   allow_nan_stats=True,
#		   name='TruncatedNormal'): 
#        parameters = dict(locals())
#        with tf.name_scope(name) as name:
#            self._loc   = tensor_util.convert_nonref_to_tensor(loc)
#            self._scale = tensor_util.convert_nonref_to_tensor(scale)
#            self._low   = tensor_util.convert_nonref_to_tensor(low)
#            self._high  = tensor_util.convert_nonref_to_tensor(high)
#            super().__init__(0., 1.)
#
#    @property
#    def loc(self):
#        return self._loc
#
#    @property
#    def scale(self):
#        return self._scale
#
#    @property
#    def low(self):
#        return self._low
#
#    @property
#    def high(self):
#        return self._high
#
#    def sample(self, *args, **kwargs):
#        loc,scale,low,high = self.loc,self.loc,self.low,self.high
#        alpha = tf.math.exp(tf.math.log((low - loc)) - tf.math.log(scale))
#        beta  = tf.math.exp(tf.math.log((low - loc)) - tf.math.log(scale))
#
#        samples = super().sample(*args, **kwargs)
#        return tf.math.ndtri(ndtr(alpha) + samples * (ndtr(beta) - ndtr(alpha))) * scale + loc
#
#
#    def log_prob(self, X):
#        
