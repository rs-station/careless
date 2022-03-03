from careless.models.base import BaseModel
from careless.utils.shame import sanitize_tensor
from careless.models.merging.surrogate_posteriors import TruncatedNormal
from tqdm.autonotebook import tqdm
import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow import keras as tfk
import numpy as np


class VariationalMergingModel(tfk.Model, BaseModel):
    """
    Merge data with a posterior parameterized by a surrogate distribution.
    """
    def __init__(self, surrogate_posterior, prior, likelihood, scaling_model, mc_sample_size=1):
        """"
        Parameters
        ----------
        surrogate_posterior : tfd.Distribution
            A surrogate posterior distribution to use. 
            If non is supplied, the default truncated normal distribution will be used. 
            Any posteriors passed in with this arg must have all properly transformed parameters in their
            `self.trainable_variables` iterable.  Use `tfp.util.TransformedVariable` to ensure positivity constraints
            where applicable.
        prior : distribution
            Prior distribution on merged, normalized structure factor amplitudes. 
            Either a Distribution from tensorflow_probability.distributions or 
            a Prior from careless.models.distributions. This must implement .log_prob. 
            This distribution must have an `event_shape` equal to `np.max(miller_ids) + 1`.
        likelihood : careless.models.likelihood.Likelihood
            This is a Likelihood object from careless.
        scaling_model : careless.models.base.BaseModel
            An instance of a class from carless.model.scaling 
        mc_sample_size : int (optional)
            This sets how many reparameterized samples will be used to compute the loss function.
        """
        super().__init__()
        self.prior = prior
        self.surrogate_posterior = surrogate_posterior
        self.likelihood = likelihood
        self.scaling_model = scaling_model
        self.mc_sample_size = mc_sample_size

    def prediction_mean_stddev(self, inputs):
        """
        Parameters
        ----------
        inputs : data
            inputs is a data structure like [refl_id, image_id, metadata, intensity, uncertainty]. 
            This can be a tf.DataSet, or a group of tensors. 

        Returns
        -------
        mean : np.array
            A numpy array containing the mean value predicted by the model for each input. 
        stddev : np.array
            A numpy array containing the standard deviation predicted by the model for each input. 
            This is a reasonable estimate of the uncertainty of the model about each input.
        """
        refl_id = self.get_refl_id(inputs)
        #Let's actually return the expected value of the data under the current model
        #This is <F**2.>
        scale_dist = self.scaling_model(inputs)
        f2 = tf.square(self.surrogate_posterior.mean()) + tf.square(self.surrogate_posterior.stddev())
        iexp = scale_dist.mean() * tf.gather(f2, tf.squeeze(refl_id, axis=-1), axis=-1)
        iexp = iexp.numpy()

        from scipy.stats import truncnorm
        q = self.surrogate_posterior
        low,high,loc,scale = q.low.numpy(),q.high.numpy(),q.loc.numpy(),q.scale.numpy()
        low,high = np.ones_like(loc)*low,np.ones_like(loc)*high
        # This moment function does not vectorize for some reason
        f4 = np.array([truncnorm.moment(4, *i) for i in zip(low, high, loc, scale)])

        s2 = np.square(scale_dist.mean().numpy()) + np.square(scale_dist.stddev().numpy())
        # var(I) = <I^2> - <I>^2
        # <I^2> = <F^4><Sigma^2>
        ivar = f4[np.squeeze(refl_id)]*s2 - iexp*iexp

        # We need to convolve the predictions if this is laue data
        from careless.models.likelihoods.laue import LaueBase
        if isinstance(self.likelihood, LaueBase):
            likelihood = self.likelihood(inputs)
            iexp = likelihood.convolve(iexp)
            ivar = likelihood.convolve(ivar)
            iexp,ivar = iexp.numpy(),ivar.numpy()

        return iexp,np.sqrt(ivar)

    def call(self, inputs):
        """
        Parameters
        ----------
        inputs : data
            inputs is a data structure like [refl_id, image_id, metadata, intensity, uncertainty]. 
            This can be a tf.DataSet, or a group of tensors. 

        Returns
        -------
        predictions : tf.Tensor
            Values predicted by the model for this sample. 
        """
        z_f = self.surrogate_posterior.sample(self.mc_sample_size)

        scale_dist = self.scaling_model(inputs)
        z_scale = scale_dist.sample(self.mc_sample_size)

        kl_div = tf.reduce_sum(self.surrogate_posterior.log_prob(z_f) - self.prior.log_prob(z_f))

        refl_id = self.get_refl_id(inputs)

        ipred = z_scale * tf.square(tf.gather(z_f, tf.squeeze(refl_id, axis=-1), axis=-1))

        likelihood = self.likelihood(inputs)

        ll = tf.reduce_sum(likelihood.log_prob(ipred))

        #just to make things consistent across mc_samples
        ll     /= self.mc_sample_size
        kl_div /= self.mc_sample_size

        #Do some keras-y stuff
        self.add_loss(-ll)
        self.add_loss(kl_div)
        self.add_metric(-ll, name="NLL")
        self.add_metric(kl_div, name="KLDiv")

        return ipred

    def train_model(self, data, steps, message=None, format_string="{:0.2e}"):
        """
        Alternative to the keras backed VariationalMergingModel.fit method. This method is much faster at the moment but less flexible.
        """
        @tf.function
        def train_step(model_and_inputs):
            model, data = model_and_inputs
            return model.train_step((data,))

        history = {}
        from tqdm import trange
        bar = trange(steps, desc=message)
        for i in bar:
            _history = train_step((self, data))
            pf = {}
            for k,v in _history.items():
                v = float(v)
                pf[k] = format_string.format(v)
                if k not in history:
                    history[k] = []
                history[k].append(v)
            bar.set_postfix(pf)
        return history
