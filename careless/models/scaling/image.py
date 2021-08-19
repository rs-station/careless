import tensorflow as tf
from careless.models.base import BaseModel
import tensorflow_probability as tfp
import numpy as np


class ImageScaler(BaseModel):
    """
    Simple linear image scales. Average value pegged at 1.
    """
    def __init__(self, max_images):
        """
        Parameters
        ----------
        max_images : int
            The maximum number of image variables to be learned
        """
        super().__init__()
        self._scales = tf.Variable(tf.ones(self.num_groups - 1))

    @property
    def scales(self):
        return tf.concat(([1.], self._scales), axis=-1)

    def call(self, inputs):
        """
        Parameters
        ----------
        inputs : list or tf.data.DataSet
            A list of tensor inputs or a DataSet in the standard 
            careless format.

        Returns
        -------
        scales : tf.Tensor(float32)
            A tensor the same shape as image_ids.
        """
        image_ids = self.get_image_id(inputs)
        w = self.scales
        return tf.gather(w, image_ids)

class HybridImageScaler(BaseModel):
    """
    A scaler that combines an `ImageScaler` with an `MLPScaler`
    """
    def __init__(self, mlp_scaler, image_scaler):
        super().__init__()
        self.mlp_scaler = mlp_scaler
        self.image_scaler = image_scaler

    def call(self, inputs):
        """
        Parameters
        ----------
        """
        q = self.mlp_scaler(inputs)
        a = self.image_scaler(inputs)
        return tfp.distributions.TransformedDistribution(
            q,
            tfp.bijectors.AffineScalar(shift=0., scale=a),
        )

