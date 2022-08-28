from careless.utils.distributions import Rice,FoldedNormal
from tensorflow_probability.python.internal.special_math import ndtr
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
from tensorflow_probability.python.internal import tensor_util
import tensorflow as tf
import numpy as np

class SurrogatePosterior(tf.keras.models.Model):
    """ The base class for learnable variational distributions over structure factor amplitudes. """
    def __init__(self, distribution, **kwargs):
        super().__init__(self, **kwargs)
        self.distribution = distribution

    def sample(self, *args, **kwargs):
        return self.distribution.sample(*args, **kwargs)

    def log_prob(self, *args, **kwargs):
        return self.distribution.log_prob(*args, **kwargs)

    def mean(self, *args, **kwargs):
        return self.distribution.mean(*args, **kwargs)

    def stddev(self, *args, **kwargs):
        return self.distribution.stddev(*args, **kwargs)

    def parameter_properties(self, *args, **kwargs):
        return self.distribution.parameter_properties(*args, **kwargs)

    def moment_4(self):
        raise NotImplementedError("The fourth moment of this distribution is not implemented yet.")

    @property
    def parameters(self):
        return self.distribution.parameters


#This is a temporary workaround for tfd.TruncatedNormal which has a bug in sampling
#2020-10-30: This should be removed if this issue is fixed: https://github.com/tensorflow/probability/issues/1149
#
#2020-11-01: On second thought, this may not be fixed unless they git rid of the current rejection sampler based 
# implementation. See https://github.com/tensorflow/probability/issues/518, for additional issues.
class TruncatedNormal(SurrogatePosterior):
    def __init__(self, loc, scale, low, high, validate_args=False, allow_nan_stats=True, name='TruncatedNormal', **kwargs):
        distribution = tfd.TruncatedNormal(loc, scale, low, high, validate_args, allow_nan_stats, name)
        super().__init__(distribution, **kwargs)

    def sample(self, *args, **kwargs):
        s = self.distribution.sample(*args, **kwargs)
        low = self.distribution.low
        return tf.maximum(low, s)

    def _tf_moment_4(self, high=None):
        from tensorflow_probability.python.internal.special_math import ndtr
        if high is None:
            high = self.distribution.high
        a,b = self.distribution.low,high
        mu,sigma = self.distribution.loc, self.distribution.scale
        z_b = (b-mu)/sigma
        z_a = (a-mu)/sigma
        
        norm = tfd.Normal(0., 1.) #Standard Normal
        if b==np.inf:
            bterm=0.
        else:
            bterm = (b*b*b + b*b*mu + b*mu*mu + sigma*sigma*(3*b + 5*mu) + mu*mu*mu)*norm.prob(z_b) 
        aterm = (a*a*a+a*a*mu+a*mu*mu+sigma*sigma*(3*a+5*mu)+mu*mu*mu)*norm.prob(z_a)

        num = bterm - aterm
        den = ndtr(z_b) - ndtr(z_a)
        return mu*mu*mu*mu + 6*mu*mu*sigma*sigma + 3*sigma*sigma*sigma*sigma- sigma*num/den

    def _scipy_moment_4(self, high):
        from scipy.stats import truncnorm
        loc,scale = self.distribution.loc, self.distribution.scale
        low = self.distribution.low.numpy()
        if high is None:
            high = self.distribution.high.numpy()
        a, b = (low - loc) / scale, (high - loc) / scale
        mom4 = truncnorm.moment(4, a, b, loc, scale)
        return mom4

    def moment_4(self, high=np.inf, method='scipy'):
        """
        Calculate the fourth moment of this distribution. This is based on the formula here: 
        https://people.smp.uq.edu.au/YoniNazarathy/teaching_projects/studentWork/EricOrjebin_TruncatedNormalMoments.pdf

        Parameters
        ----------
        high : float (optional)
            The high parameter to use for the distribution. By default use inf.
        method : str (optional)
            Either 'scipy' or 'tf'
        """
        if method=='scipy':
            return self._scipy_moment_4(high)
        elif method == 'tf':
            return self._tf_moment_4(high)
        else:
            raise ValueError(f"Unknown method {method} for computing moment_4")

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
            tfb.Exp(),
        )
        scale = tfp.util.TransformedVariable(
            scale,
            tfb.Chain([
                tfb.Exp(),
                tfb.Shift(scale_shift),
            ]),
        )
        return cls(loc, scale, low, high)

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
        self._centric = np.array(centric, dtype=bool)
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



