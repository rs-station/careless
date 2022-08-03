from careless.models.base import BaseModel
from tensorflow import keras as tfk


class Scaler(tfk.models.Model, BaseModel):
    """ Base class for scaling models """
