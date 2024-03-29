from careless.models.base import BaseModel
import tf_keras as tfk


class Scaler(tfk.models.Model, BaseModel):
    """ Base class for scaling models """
