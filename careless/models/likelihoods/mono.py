from careless.models.likelihoods.base import Likelihood
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np

class NormalLikelihood(Likelihood):
    def call(inputs):
        return tfd.Normal(
            self.get_intensities,
            self.get_uncertainties,
        )

class LaplaceLikelihood(Likelihood):
    def call(inputs):
        return tfd.Laplace(
            self.get_intensities,
            self.get_uncertainties,
        )

class StudentTLikelihood(MonoBase):
    def __init__(self, dof):
        """
        Parameters
        ----------
        dof : float
            Degrees of freedom of the student t likelihood.
        """
        super().__init__()
        self.dof = dof

    def call(inputs):
        return tfd.StudentT(
            self.dof,
            self.get_intensities,
            self.get_uncertainties,
        )

