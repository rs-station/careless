from careless.models.base import PerGroupModel
from careless.utils.shame import sanitize_tensor
from careless.models.merging.surrogate_posteriors import TruncatedNormal
from tqdm.autonotebook import tqdm
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np


class VariationalMergingModel(PerGroupModel, tf.Module):
    """
    Merge data with a posterior parameterized by a surrogate distribution.
    """
    def __init__(self, miller_ids, scaling_models, prior, likelihood, surrogate_posterior=None, use_weighted_kl_div=False):
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
        use_weighted_kl_div : bool (optional)
            If True, the Dkl(surrogate_posterior || prior) will be weighted by the multiplicity of each reflection.
            This may lead to a better scaling model in certain cases, but by default this is set to false.
        """
        super().__init__(miller_ids)

        self.eps = 1e-12
        self.prior = prior
        self.likelihood = likelihood
        self.scaling_models = scaling_models if isinstance(scaling_models, (list, tuple)) else (scaling_models, )
        self.weight_kl = use_weighted_kl_div

        if surrogate_posterior is None:
            zero = 0.
            infinity = 1e30
            from careless.utils.distributions import FoldedNormal
            self.surrogate_posterior = TruncatedNormal(
                tf.Variable(self.prior.mean()),
                tfp.util.TransformedVariable(self.prior.stddev(), tfp.bijectors.Softplus()),
                zero,
                infinity,
            )
        else:
            self.surrogate_posterior = surrogate_posterior

        # Cache the initial values of the surrogate posterior in case they must be rescued later
        self._surrogate_posterior_init = [x.value() for x in self.surrogate_posterior.trainable_variables]

    def predict(self):
        """
        Compute the expected value of reflection observations under the current model.
        """
        scale = 1.
        for model in self.scaling_models:
            if hasattr(model, "loc"):
                scale *= model.loc
            else:
                scale *= model()

        #This is <F**2.>
        Fsquared = self.surrogate_posterior.mean()**2. + self.surrogate_posterior.stddev()**2.
        I = self.expand(Fsquared)*scale
        return I

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
            q_F = self.surrogate_posterior.log_prob(F)
            p_F = self.prior.log_prob(F)
            if self.weight_kl:
                kl_div += tf.reduce_mean(self.expand(q_F - p_F)) #<--this is abismal model
            else:
                kl_div += tf.reduce_sum( q_F - p_F ) 

        scale = 1.
        for model in self.scaling_models:
            if return_kl_term:
                sample, kl_term = model.sample(return_kl_term, sample_shape)
                kl_div += kl_term
            else: 
                sample = model.sample(return_kl_term, sample_shape)

            scale = scale*sample

        I = self.expand(tf.square(F)) * scale

        if return_kl_term:
            return I,kl_div
        else:
            return I

    @property
    def num_params(self):
        return tf.reduce_sum([tf.reduce_prod(i.shape) for i in self.trainable_variables])

    def akaike_information_criterion(self, samples=10):
        log_prob = 0
        for i in range(samples):
            I,kl_div = self.sample(return_kl_term=True)
            log_prob += tf.reduce_sum(self.likelihood.log_prob(I))
        return 2*tf.math.log(tf.cast(self.num_params, tf.float32)) - 2.*log_prob/samples

    def bayesian_information_criterion(self, samples=10):
        log_prob = 0
        for i in range(samples):
            I,kl_div = self.sample(return_kl_term=True)
            log_prob += tf.reduce_sum(self.likelihood.log_prob(I))
        return tf.math.log(tf.cast(self.num_params, tf.float32))*self.num_observations - 2.*log_prob/samples

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
        log_prob = self.likelihood.log_prob(I)
        if self.weight_kl:
            log_likelihood = tf.reduce_mean(log_prob)
        else:
            log_likelihood = tf.reduce_sum(log_prob)
        loss = -log_likelihood + kl_div
        return loss

    def loss_and_grads(self, variables, s=1):
        with tf.GradientTape() as tape:
            loss = 0.
            for i in range(s):
                loss += self()/s
        grads = tape.gradient(loss, variables)
        return loss, grads

    def _train_step(self, optimizer, s=1, clip_value=None):
        variables = self.trainable_variables
        loss, grads = self.loss_and_grads(variables, s)
        #Occasionally, low probability samples will lead to overflows in the gradients. 
        #Since, VI is usually done by coordinate ascent anyway, 
        #it is totally fine to just skip those updates.
        nans = tf.reduce_any([tf.reduce_any(tf.math.is_nan(g)) for g in grads])
        if nans:
            tf.print("Warning: Nans in grads. This is expected to happen occasionally. Sanitizing...")
            grads = tf.nest.map_structure(sanitize_tensor, grads)

        optimizer.apply_gradients(zip(grads, variables))
        return loss

    @tf.function
    def train_step(self, optimizer, s=1, clip_value=None):
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
        return self._train_step(optimizer, s=s, clip_value=clip_value)

    def rescue_variational_distributions(self):
        """
        Reset values for problem distributions to their intial values. 
        This is useful if probabilities posteriors get stuck in a low probability region and samples underflow. 
        They can also get stuck in a regime where the stddev overflows. 
        This method corrects that pathology as well.
        """
        F = self.surrogate_posterior.sample()
        q_F = self.surrogate_posterior.prob(F)
        #This is useful if some of the posteriors get stuck and probabilities underflow
        for initial_value,var in zip(self._surrogate_posterior_init, self.surrogate_posterior.trainable_variables):
            var.assign(tf.where(tf.math.is_finite(q_F), var, initial_value))

        #This is useful if some of the posteriors get stuck and stddev overflows
        sigma = self.surrogate_posterior.stddev()
        for initial_value,var in zip(self._surrogate_posterior_init, self.surrogate_posterior.trainable_variables):
            var.assign(tf.where(tf.math.is_finite(sigma), var, initial_value))

    def fit(self, optimizer=None, iterations=10000, max_nancount=20, s=1, clip_value=None):
        """
        Fit the model by making the specified number of optimizer steps.

        Parameters
        ----------
        optimizer : tf.keras.optimizers.Optimizer
            Keras style optimizer. If none is supplied, an Adam optimizer with learning rate 0.001 will be used.
        iterations : int
            Number of gradient steps to make. The default is 10,000
        max_nancount : int
            Maximum number of successive NaN valued losses before terminating opimization. The Default is 20. 
            This rarely matters with the current implementation, but it might be important for tricky likelihoods.
        s : int
            Number of samples used for gradient estimates. The Default is 1.

        Returns
        -------
        losses : ndarray
            Numpy array containing the loss function value for each gradient step.
        """
        if optimizer is None:
            optimzer = tf.keras.optimizers.Adam(0.001)

        losses = []
        print(f"{'#'*80}")
        print(f'Optimizing model')
        print(f"{'#'*80}")
        loss_ = None
        chol_fails = 0
        nancount = 0
        for _ in tqdm(range(iterations)):
            loss = self.train_step(optimizer, s=s, clip_value=clip_value)
            losses.append(float(loss))
            if not tf.math.is_finite(loss):
                #self.rescue_variational_distributions()
                #print(f"WARNING! Resetting stuck variational distributions!")
                nancount += 1
            else:
                nancount = 0
            loss_ = loss

            if nancount > max_nancount:
                print(f"WARNING! Optimization terminated due to too many failed gradient steps. Try decreasing the learning rate.")
                break

        return np.array(losses)

