from careless.models.base import PerGroupModel
from careless.utils.shame import sanitize_tensor
from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np


class VariationalMergingModel(PerGroupModel):
    """
    Merge data with a posterior parameterized by a surrogate distribution.
    """
    def __init__(self, miller_ids, scaling_models, prior, likelihood, surrogate_posterior=None):
        """"
        Parameters
        ----------
        miller_ids : array
            Numpy array or tf.Tensor with zero indexed miller ids that map each reflection observation to a 
            miller index. In the case of a harmonic likelihood function, this array may be longer than actual
            number of observed reflection intensities. 
        scaling_models : careless.models.scaling.Scaler or iterable
            An instance of a carless.model.scaling.Scaler class or list/tuple thereof.
        prior : distribution
            Prior distribution on merged, normalized structure factor amplitudes. 
            Either a Distribution from tensorflow_probability.distributions or 
            a Prior from careless.models.distributions. This must implement .prob and .log_prob. 
            This distribution must have an `event_shape` equal to `np.max(miller_ids) + 1`.
        posterior_truncated_normal_max : float
            the maximum value of the surrogate posterior distribution. 
            this defaults to 1e10. this should only need to be changed if you are using a prior on an empirical scale.
        likelihood : distribution
            A distribution with a `log_prob` and `prob` method. 
            These methods must accept a `x : tf.Tensor` with `len(x) == len(miller_ids)`. 
            For harmonic likelihoods, it may be the case the returned vector is smaller than `len(miller_ids)`.
        surrogate_posterior : tfd.Distribution
            A surrogate posterior distribution to use. 
            If non is supplied, the default truncated normal distribution will be used. 
            Any posteriors passed in with this arg must have all properly transformed parameters in their
            `self.trainable_variables` iterable.  Use `tfp.util.TransformedVariable` to ensure positivity constraints
            where applicable.
        """
        super().__init__(miller_ids)
        self.prior = prior
        self.likelihood = likelihood
        self.scaling_models = scaling_models if isinstance(scaling_models, (list, tuple)) else (scaling_models, )

        if surrogate_posterior is None:
            self.surrogate_posterior = tfd.TruncatedNormal(
                tf.Variable(self.prior.mean(), dtype=tf.float32),
                tfp.util.TransformedVariable(self.prior.stddev(), tfp.bijectors.Softplus()),
                0., 
                1e10,
            )
        else:
            self.surrogate_posterior = surrogate_posterior

        # Cache the initial values of the surrogate posterior in case they must be rescued later
        self._surrogate_posterior_init = [x.value() for x in self.surrogate_posterior.trainable_variables]

        # Put the trainable_variables at the top level
        self.trainable_variables  = self.surrogate_posterior.trainable_variables 
        for i,model in enumerate(self.scaling_models):
            if isinstance(model.trainable_variables, (tuple, list)):
                self.trainable_variables += tuple(model.trainable_variables )
            else:
                typename = type(model.trainable_variables)
                raise TypeError(f"scaling_models[{i}].trainable_variables has type {typename} but only tuple or list allowed")

    def sample(self, return_kl_term=False, sample_shape=(), seed=None, name='sample', **kwargs):
        """
        Randomly sample predicted reflection observations.

        Parameters
        ----------
        return_kl_term : bool
            if `True` this function returns a tuple of `tf.Tensor` instances, `(sample, kl_term)`, wherein `kl_term` 
            is the Kullback-Leibler divergence between the variational distribution and its prior plus the 
            divergence between the scales and any priors that may be placed on them.
        sample_shape : int
            Shape of returned samples. Defaults to () for a single sample. 
        seed : int
            Defaults to None.
        name : str
            Defaults to 'sample'

        Returns
        -------
        sample or (sample, kl_term) : tf.Tensor
            Either a sample of the predicted reflections intensities or a sample and corresponding kl_div.
        """
        F = self.surrogate_posterior.sample(sample_shape, seed, name, **kwargs)

        kl_div = 0.
        if return_kl_term:
            q_F = self.surrogate_posterior.prob(F)
            p_F = self.prior.prob(F)
            kl_div += tf.reduce_sum( q_F * ( tf.math.log(q_F) - tf.math.log(p_F) ) )

        scale = 1.
        for model in self.scaling_models:
            if return_kl_term:
                sample, kl_term = model.sample(return_kl_term, sample_shape)
                kl_div += kl_term
            else: 
                sample = model.sample(return_kl_term, sample_shape)

            scale = scale*sample

        I = self.expand(F**2.) * scale

        if return_kl_term:
            return I,kl_div
        else:
            return I

    def __call__(self, sample_shape=()):
        """
        Parameters
        ----------
        sample_shape : () or int
            () for a single sample or an integer number of samples. More complex sample shapes are not supported.

        Returns
        -------
        loss : Tensor
            The scalar value of the Evidence Lower BOund.
        """
        I,kl_div = self.sample(return_kl_term=True, sample_shape=sample_shape)
        log_likelihood = tf.reduce_sum(self.likelihood.log_prob(I))
        loss = -log_likelihood + kl_div
        return loss

    def loss_and_grads(self, variables, s=1):
        with tf.GradientTape() as tape:
            loss = 0.
            for i in range(s):
                loss += self()/s
        grads = tape.gradient(loss, variables)
        return loss, grads

    def _train_step(self, optimizer, s=1):
        variables = self.trainable_variables
        loss, grads = self.loss_and_grads(variables, s)
        #Occasionally, low probability samples will lead to overflows in the gradients. 
        #Since, VI is usually done by coordinate ascent anyway, 
        #it is totally fine to just skip those updates.
        grads = [sanitize_tensor(g) for g in grads]
        optimizer.apply_gradients(zip(grads, variables))
        return loss

    @tf.function
    def train_step(self, optimizer, s=1):
        """
        Parameters
        ----------
        optimizer : tf.keras.optimizer.Optimizer
            A keras style optimizer
        Returns
        -------
        loss : float
            The current value of the Evidence Lower BOund
        """
        return self._train_step(optimizer, s=1)

    def rescue_variational_distributions(self):
        """
        Reset values for problem distributions to their intial values. 
        This is useful if probabilities posteriors get stuck in a low probability region and samples underflow. 
        """
        F = self.surrogate_posterior.sample()
        q_F = self.surrogate_posterior.prob(F)
        #This is useful if some of the posteriors get stuck
        for initial_value,var in zip(self._surrogate_posterior_init, self.surrogate_posterior.trainable_variables):
            var.assign(tf.where(tf.math.is_finite(q_F), var, initial_value))

