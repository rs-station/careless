import tensorflow as tf
import tensorflow_probability as tfp
from careless.models.base import PerGroupModel
from careless.models.scaling.base import Scaler
import numpy as np


class ImageScaler(PerGroupModel, Scaler, tf.Module):
    """
    Simple linear image scales. Average value pegged at 1.
    """
    def __init__(self, image_number):
        """
        Parameters
        ----------
        image_number : array 
            array of zero indexed image ids. One for each reflection observation.
        """
        image_number = tf.convert_to_tensor(image_number, dtype=tf.int64)
        super().__init__(image_number)

        self._scales = tf.Variable(tf.ones(self.num_groups - 1))

    @property
    def scales(self):
        return tf.concat(([1.], self._scales), axis=-1)

    def sample(self, return_kl_term=False, *args, **kwargs):
        """ This is not a real distribution per se. """
        w = self.expand(self.scales)
        if return_kl_term:
            return w / tf.reduce_mean(w) , 0.
        else:
            return w / tf.reduce_mean(w) 

class VariationalImageScaler(ImageScaler):
    """
    Variational image scaling model with a prior distribution for scales. 
    """
    def __init__(self, image_number, prior, surrogate_posterior=None):
        """
        Parameters
        ----------
        image_number : array 
            array of zero indexed image ids. One for each reflection observation.
        prior : tfd.Distribution 
            tensorflow probability distribution with `batch_shape == image_number.max() + 1`
        surrogate_posterior : tfd.Distribution (optional)
            tensorflow probability distribution with `batch_shape == image_number.max() + 1`.
            If None, a Normal surrogate will be initialized as:
            ```
            self.surrogate_posterior = tfd.Normal(prior.mean(), prior.stddev())
            ```
        """
        image_number = tf.convert_to_tensor(image_number, dtype=tf.int64)
        super().__init__(image_number)

        self.prior = prior
        self._scales = tf.Variable(tf.ones(self.num_groups - 1))
        if surrogate_posterior is None:
            loc = tf.Variable(self.prior.mean()),
            scale = tfp.util.TransformedVariable(
                tf.Variable(self.prior.stddev()),
                tfp.bijectors.Softplus(),
            )
            self.surrogate_posterior = tfp.distributions.Normal(loc, scale)
        else:
            self.surrogate_posterior = surrogate_posterior

    def sample(self, return_kl_term=False, *args, **kwargs):
        """ 
        This will return a sample of the variational image weights. 
        Right now, this will use the slower gather operation instead of the sparse
        matmul for expansion. This is in order to support arbitrary sample sizes.
        """
        z = self.surrogate_posterior.sample(*args, **kwargs)
        if return_kl_term:
            log_q = self.surrogate_posterior.log_prob(z)
            log_p = self.prior.log_prob(z)
            return self.gather(z, self.group_ids),tf.reduce_sum(log_q - log_p)
        else:
            return self.gather(z, self.group_ids)

