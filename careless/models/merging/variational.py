from careless.models.base import BaseModel
from careless.utils.shame import sanitize_tensor
from careless.models.merging.surrogate_posteriors import TruncatedNormal
import tf_keras as tfk
from tqdm.autonotebook import tqdm
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np


class VariationalMergingModel(tfk.models.Model, BaseModel):
    """
    Merge data with a posterior parameterized by a surrogate distribution.
    """
    def __init__(self, surrogate_posterior, prior, likelihood, scaling_model, mc_sample_size=1, kl_weight=None, scale_kl_weight=None, scale_prior=None):
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
        self.kl_weight = kl_weight
        self.scale_kl_weight = scale_kl_weight
        self.scale_prior = scale_prior

    def scale_mean_stddev(self, inputs):
        """
        Compute the moments of the posterior of reflection observation scale factors. 

        Parameters
        ----------
        inputs : data
            inputs is a data structure like [refl_id, image_id, metadata, intensity, uncertainty]. 
            This can be a tf.DataSet, or a group of tensors. 

        Returns
        -------
        mean : np.array
            A numpy array containing the mean value of the scale predicted by the model for each input. 
        stddev : np.array
            A numpy array containing the standard deviation of the scale predicted by the model for each input. 
            This is a reasonable estimate of the uncertainty of the model about each input.
        """
        refl_id = self.get_refl_id(inputs)

        scale_dist = self.scaling_model(inputs)
        mean = scale_dist.mean().numpy()
        stddev = scale_dist.stddev().numpy()

        # We need to convolve the predictions if this is laue data
        from careless.models.likelihoods.laue import LaueBase
        if isinstance(self.likelihood, LaueBase):
            likelihood = self.likelihood(inputs)
            mean = likelihood.convolve(mean)
            stddev = np.sqrt(likelihood.convolve(stddev * stddev))

        return mean, stddev

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
        f4 = q.moment_4(method='scipy')

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

    def add_kl_div(self, posterior, prior, samples=None, weight=1., reduction='sum', name="KLDiv"):
        try:
            kl_div = posterior.kl_divergence(prior)
        except:
            NotImplementedError
            kl_div = posterior.log_prob(samples) - prior.log_prob(samples)

        if reduction == 'sum':
            kl_div = tf.reduce_sum(kl_div) / self.mc_sample_size
        elif reduction == 'mean':
            kl_div = tf.reduce_mean(kl_div) 
        else:
            kl_div = reduction(kl_div)

        self.add_loss(weight * kl_div)
        self.add_metric(kl_div, name=name)
        return kl_div

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

        if self.scale_prior is not None:
            if self.scale_kl_weight is None:
                self.add_kl_div(scale_dist, self.scale_prior, z_scale, weight=self.scale_kl_weight, reduction='sum', name="Σ KLDiv")
            else:
                self.add_kl_div(scale_dist, self.scale_prior, z_scale, weight=1., reduction='mean', name="Σ KLDiv")

        refl_id = self.get_refl_id(inputs)

        ipred = z_scale * tf.square(tf.gather(z_f, tf.squeeze(refl_id, axis=-1), axis=-1))

        likelihood = self.likelihood(inputs)

        ll = likelihood.log_prob(ipred)
        if self.kl_weight is None:
            self.add_kl_div(self.surrogate_posterior, self.prior, z_f, name='F KLDiv', reduction='sum')
            ll = tf.reduce_sum(ll) / self.mc_sample_size
        else:
            self.add_kl_div(self.surrogate_posterior, self.prior, z_f, weight=self.kl_weight, name='F KLDiv', reduction='mean')
            ll = tf.reduce_mean(ll) 

        #Do some keras-y stuff
        self.add_loss(-ll)
        self.add_metric(-ll, name="NLL")

        return ipred

    def train_step_with_gradient_norm(self, data=None):
        """
        Conduct a training step with `data`. This method is the same as tfk.Model.train_step except that it
        tracks the norm of the gradients as well. 
        """
        if data is None:
            x = self.data
        else:
            x = data[0]
        y = self.get_intensities(x)

        # Run forward pass.
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Run backwards pass.
        grads = tape.gradient(loss, self.trainable_variables)

        # Compute the L2 norm of the gradients
        grad_norm = tf.linalg.global_norm(grads)

        # Only apply gradients if they are valid
        grads = [tf.where(tf.math.is_finite(g), g, 0.) for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.compiled_metrics.update_state(y, y_pred)

        # Collect metrics to return
        return_metrics = {
            "Grad Norm" : grad_norm,
        }
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result

        return return_metrics

    def train_model(self, data, steps, message=None, format_string="{:0.2e}", validation_data=None, validation_frequency=10, progress=True, use_custom_train_step=True, jit_compile=None, reduce_retracing=False):
        """
        Alternative to the keras backed VariationalMergingModel.fit method. This method is much faster at the moment but less flexible.
        """
        if use_custom_train_step:
            def train_step(model_and_data):
                model, data = model_and_data
                model.reset_metrics()
                history = model.train_step_with_gradient_norm((data,))
                return history
        else:
            def train_step(model_and_data):
                model, data = model_and_data
                model.reset_metrics()
                history = model.train_step((data,))
                return history

        if not self._run_eagerly:
            train_step = tf.function(
                train_step, reduce_retracing=reduce_retracing, jit_compile=jit_compile
            )

        if validation_data is not None:
            val_scale = len(data[0]) / len(validation_data[0])

        history = {}
        from tqdm import trange
        disable_progress = not progress
        bar = trange(steps, desc=message, disable=disable_progress)
        for i in bar:
            _history = train_step((self, data))
            if validation_data is not None:
                if i%validation_frequency==0:
                    validation_metrics = self.test_on_batch(validation_data, return_dict=True)
                _history['NLL_val'] = val_scale * validation_metrics['NLL']

            pf = {}
            for k,v in _history.items():
                v = float(v)
                pf[k] = format_string.format(v)
                if k not in history:
                    history[k] = []
                history[k].append(v)

            bar.set_postfix(pf)
            if use_custom_train_step:
                if not tf.math.is_finite(_history['Grad Norm']):
                    print("Encountered numerical issues, terminating optimization early!")
                    break
        return history

