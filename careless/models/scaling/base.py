from careless.models.base import BaseModel
import tf_keras as tfk
import tensorflow as tf
from tensorflow_probability import distributions as tfd


class Scaler(tfk.models.Model, BaseModel):
    """ Base class for scaling models """

class ConstantScaler(Scaler):
    """
    A scaler that learns a single global scaling factor for the entire dataset.
    Outputs a deterministic distribution with zero variance.
    """
    def __init__(self, initial_value=1.0, scale_bijector=None):
        """
        Parameters
        ----------
        initial_value : float
            The initial value for the global scale (e.g., intensity standard deviation).
        scale_bijector : tfp.bijectors.Bijector
            Bijector to map the unconstrained variable to the positive real line.
        """
        super().__init__()
        if scale_bijector is None:
            scale_bijector = tfb.Exp()
        self.scale_bijector = scale_bijector

        # Initialize the variable in the unconstrained space
        init_tensor = tf.constant(initial_value, dtype=tf.float32)
        unconstrained_init = self.scale_bijector.inverse(init_tensor)

        self.w = tf.Variable(unconstrained_init, name='global_scale', trainable=True)

    def call(self, inputs):
        """
        Returns a Deterministic distribution centered at the learned global scale.
        """
        refl_id = self.get_refl_id(inputs)

        refl_id = tf.squeeze(refl_id, axis=-1) 

        # Transform variable to positive scale
        scale = self.scale_bijector(self.w)

        # Broadcast to shape of inputs (Batch,)
        loc = scale * tf.ones_like(refl_id, dtype=tf.float32)

        return tfd.Deterministic(loc=loc)
