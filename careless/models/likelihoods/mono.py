import torch
import torch.nn as nn
import math
import numpy as np
from torch.distributions import Normal, Laplace
from careless.distributions import StudentT
from rs_distributions.modules import TransformedParameter
from torch.distributions.transforms import SoftplusTransform
from careless.models.likelihoods.base import Likelihood


class LocationScaleLikelihood(Likelihood):
    def get_loc_and_scale(self, inputs):
        loc = self.get_intensities(inputs).squeeze(-1).float()
        scale = self.get_uncertainties(inputs).squeeze(-1).float()
        return loc, scale


class NormalLikelihood(LocationScaleLikelihood):
    def forward(self, inputs):
        return Normal(*self.get_loc_and_scale(inputs))


class LaplaceLikelihood(LocationScaleLikelihood):
    def forward(self, inputs):
        loc, scale = self.get_loc_and_scale(inputs)
        return Laplace(loc, scale / math.sqrt(2.0))


class StudentTLikelihood(LocationScaleLikelihood):
    def __init__(self, dof):
        """
        Parameters
        ----------
        dof : float
            Degrees of freedom of the Student-T likelihood.
        """
        super().__init__()
        self.dof = float(dof)

    def forward(self, inputs):
        return StudentT(self.dof, *self.get_loc_and_scale(inputs))


class Ev11Likelihood(LocationScaleLikelihood):
    """
    Error-correction likelihood following the SCALA/XDS ev11 model.
    Refines observation uncertainties as a function of predicted intensity.
    """

    def __init__(self):
        super().__init__()
        # All three parameters are strictly positive
        softplus = SoftplusTransform()
        self.Sdfac = TransformedParameter(torch.ones(1), softplus)
        self.Sdadd = TransformedParameter(torch.ones(1), softplus)
        self.SdB   = TransformedParameter(torch.ones(1), softplus)
        self._loc = None
        self._scale = None

    def forward(self, inputs):
        self._loc, self._scale = self.get_loc_and_scale(inputs)
        return self  # acts as its own distribution wrapper

    def corrected_sigiobs(self, ipred):
        ipred = torch.nn.functional.softplus(ipred)
        sigiobs = self.Sdfac() * torch.sqrt(
            self._scale ** 2
            + self.SdB() * ipred
            + self.Sdadd() * ipred ** 2
        )
        return sigiobs


class NormalEv11Likelihood(Ev11Likelihood):
    def log_prob(self, ipred):
        scale = self.corrected_sigiobs(ipred)
        return Normal(self._loc, scale).log_prob(ipred)


class StudentTEv11Likelihood(Ev11Likelihood):
    def __init__(self, dof):
        super().__init__()
        self.dof = float(dof)

    def log_prob(self, ipred):
        scale = self.corrected_sigiobs(ipred)
        return StudentT(self.dof, self._loc, scale).log_prob(ipred)


class NeuralLikelihood(Likelihood):
    """Neural network that refines observation uncertainties."""

    def __init__(self, mlp_layers, mlp_width):
        super().__init__()
        layers = []
        for _ in range(mlp_layers):
            layers.append(nn.Linear(2 if len(layers) == 0 else mlp_width, mlp_width))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(mlp_width if mlp_layers > 0 else 2, 1))
        layers.append(nn.Softplus())
        self.network = nn.Sequential(*layers)

    def base_dist(self, loc, scale):
        raise NotImplementedError("Subclasses must implement base_dist(loc, scale)")

    def forward(self, inputs):
        iobs = self.get_intensities(inputs).float()
        sigiobs = self.get_uncertainties(inputs).float()
        delta = self.network(torch.cat([iobs, sigiobs], dim=-1))
        sigpred = sigiobs * delta / delta.mean().clamp(min=1e-10)
        return self.base_dist(iobs.squeeze(-1), sigpred.squeeze(-1))


class NeuralNormalLikelihood(NeuralLikelihood):
    def base_dist(self, loc, scale):
        return Normal(loc, scale)
