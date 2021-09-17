from careless.utils.distributions import Rice,FoldedNormal
from tensorflow_probability.python.internal.special_math import ndtr
import tensorflow_probability as tfp
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

    @classmethod
    def from_loc_and_scale(cls, loc, scale, low=0., high=1e10, scale_shift=1e-7):
        """
        Instantiate a learnable distribution with good default bijectors.

        loc : array
            The initial location of the distribution
        scale : array
            The initial scale parameter of the distribution
        low : float or array (optional)
            The lower limit of the support for the distribution.
        high : float or array (optional)
            The upper limit of the support for the distribution.
        scale_shift : float (optional)
            A small constant added to the scale to increase numerical stability.
        """
        loc   = tfp.util.TransformedVariable(
            loc,
            tfb.Softplus(),
        )
        scale = tfp.util.TransformedVariable(
            scale,
            tfb.Chain([
                tfb.Softplus(),
                tfb.Shift(scale_shift),
            ]),
        )
        return cls(loc, scale, low, high)


