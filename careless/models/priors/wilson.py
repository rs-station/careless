import torch
import torch.nn as nn
import numpy as np
import reciprocalspaceship as rs
from careless.models.priors.base import Prior
from rs_distributions.modules import TransformedParameter
from torch.distributions import HalfNormal, Weibull, Normal, constraints
from torch.distributions.transforms import SigmoidTransform


class Centric(HalfNormal):
    """Half-normal Wilson prior for centric reflections."""
    def __init__(self, epsilon, sigma=1.):
        epsilon = torch.as_tensor(epsilon, dtype=torch.float32)
        sigma = torch.as_tensor(sigma, dtype=torch.float32)
        super().__init__(torch.sqrt(epsilon * sigma))

    @classmethod
    def _from_scale(cls, scale):
        obj = cls.__new__(cls)
        HalfNormal.__init__(obj, scale)
        return obj


class Acentric(Weibull):
    """Weibull Wilson prior for acentric reflections."""
    def __init__(self, epsilon, sigma=1.):
        epsilon = torch.as_tensor(epsilon, dtype=torch.float32)
        sigma = torch.as_tensor(sigma, dtype=torch.float32)
        super().__init__(
            torch.sqrt(epsilon * sigma),
            torch.tensor(2.0),
        )

    @classmethod
    def _from_scale(cls, scale):
        obj = cls.__new__(cls)
        Weibull.__init__(obj, scale, torch.full_like(scale, 2.0))
        return obj


class WilsonPrior(Prior):
    """Wilson's priors on structure factor amplitudes."""

    def __init__(self, centric, epsilon, sigma=1.):
        """
        Parameters
        ----------
        centric : array
            Boolean array; True for centric reflections.
        epsilon : array
            Multiplicity values for each structure factor.
        sigma : float or array
            Wilson distribution scale (average intensity by resolution).
        """
        super().__init__()
        epsilon = np.array(epsilon, dtype=np.float32)
        centric_bool = np.array(centric, dtype=bool)
        sigma = np.array(sigma, dtype=np.float32) * np.ones_like(epsilon)

        # Store distribution parameters as buffers so .to(device) moves them.
        self.register_buffer('centric', torch.from_numpy(centric_bool))
        self.register_buffer('_scale', torch.from_numpy(np.sqrt(epsilon * sigma)))

    @property
    def p_centric(self):
        return Centric._from_scale(self._scale)

    @property
    def p_acentric(self):
        return Acentric._from_scale(self._scale)

    def log_prob(self, x):
        log_p_c = self.p_centric.log_prob(x)
        log_p_a = self.p_acentric.log_prob(x)
        return torch.where(self.centric, log_p_c, log_p_a)

    def prob(self, x):
        return self.log_prob(x).exp()

    @property
    def mean(self):
        return torch.where(self.centric, self.p_centric.mean, self.p_acentric.mean)

    @property
    def stddev(self):
        var_c = self.p_centric.variance
        var_a = self.p_acentric.variance
        return torch.where(self.centric, var_c.sqrt(), var_a.sqrt())

    def sample(self, sample_shape=torch.Size()):
        s_c = self.p_centric.sample(sample_shape)
        s_a = self.p_acentric.sample(sample_shape)
        return torch.where(self.centric, s_c, s_a)


class DoubleWilsonPrior(Prior):
    """
    Double Wilson prior: a multivariate prior that couples related ASUs.
    Root ASUs use a WilsonPrior; child ASUs use a conditional Rice/Woolfson prior.
    """

    def __init__(self, asu_collection, parents, r_values, reindexing_ops=None,
                 sigma=1., optimize_r=False):
        """
        Parameters
        ----------
        asu_collection : ReciprocalASUCollection
        parents : list
            parents[i] = j means asu_id==i has parent asu_id==j; None for roots.
        r_values : list or array
            Correlation coefficient per child ASU.
        reindexing_ops : list, optional
            gemmi.Op instances for reindexing child→parent.
        sigma : float or array
            Wilson scale parameter.
        optimize_r : bool
            Whether to allow r to be optimized.
        """
        super().__init__()
        from rs_distributions.distributions import Rice, FoldedNormal

        self.parents = parents
        self.optimize_r = optimize_r

        r_tensor = torch.as_tensor(r_values, dtype=torch.float32)
        if optimize_r:
            self._r = TransformedParameter(r_tensor, SigmoidTransform())
        else:
            self.register_buffer('_r_fixed', r_tensor)

        reflids_list = []
        root_list = []

        for child, parent in enumerate(parents):
            child_asu = asu_collection.reciprocal_asus[child]

            if parent is None:
                reflids_list.append(child_asu.lookup_table.id.to_numpy('int32'))
                root_list.append(np.ones(len(child_asu.lookup_table), dtype=bool))
            else:
                root_list.append(np.zeros(len(child_asu.lookup_table), dtype=bool))
                parent_asu = asu_collection.reciprocal_asus[parent]
                h = child_asu.Hall
                if reindexing_ops is not None:
                    op = reindexing_ops[child]
                    h = rs.utils.apply_to_hkl(h, op)
                h, _ = rs.utils.hkl_to_asu(h, parent_asu.spacegroup)
                pid = parent * np.ones((len(h), 1), dtype='int32')
                reflids_list.append(
                    asu_collection.to_refl_id(pid, h, allow_missing=True)
                )

        centric = np.array(asu_collection.centric, dtype=bool)
        multiplicity = np.array(asu_collection.multiplicity, dtype=np.float32)
        asu_ids = np.array(asu_collection.asu_ids, dtype='int64')
        reflids = np.concatenate(reflids_list)
        root = np.concatenate(root_list)

        self.register_buffer('centric', torch.from_numpy(centric))
        self.register_buffer('multiplicity', torch.from_numpy(multiplicity))
        self.register_buffer('asu_ids', torch.from_numpy(asu_ids))
        self.register_buffer('reflids', torch.from_numpy(reflids.astype('int64')))
        self.register_buffer('absent', torch.from_numpy(reflids == -1))
        self.register_buffer('root', torch.from_numpy(root))

        sigma_arr = np.array(sigma, dtype=np.float32) * np.ones(len(centric), dtype=np.float32)
        self.wilson_prior = WilsonPrior(centric, multiplicity, sigma_arr)
        self.sigma = sigma

    @property
    def r(self):
        if self.optimize_r:
            return self._r()
        return self._r_fixed

    @property
    def mean(self):
        return self.wilson_prior.mean

    @property
    def stddev(self):
        return self.wilson_prior.stddev

    def log_prob(self, z):
        from rs_distributions.distributions import Rice, FoldedNormal

        r = torch.gather(self.r, 0, self.asu_ids)

        mask = self.reflids >= 0
        safe_reflids = torch.where(mask, self.reflids, torch.zeros_like(self.reflids))

        # Gather parent structure factors
        z_parent = torch.where(
            mask.unsqueeze(0),
            z[..., safe_reflids],
            torch.zeros_like(z[..., safe_reflids]),
        )

        loc = torch.where(self.absent, torch.zeros_like(z_parent), z_parent * r)
        r2 = r ** 2

        scale = torch.where(
            self.centric,
            torch.sqrt(self.multiplicity * self.sigma * (1.0 - r2)),
            torch.sqrt(0.5 * self.multiplicity * self.sigma * (1.0 - r2)),
        )

        # Rice (acentric) / FoldedNormal (centric) conditional prior
        rice = Rice(loc.abs(), scale)
        folded = FoldedNormal(loc, scale)
        log_p_dw = torch.where(self.centric, folded.log_prob(z), rice.log_prob(z))

        p_wilson = self.wilson_prior.log_prob(z)
        log_p = torch.where(self.root, p_wilson, log_p_dw)

        # Log individual r values as metrics
        for i, r_val in enumerate(self.r.unbind()):
            self.add_metric(r_val, f"rDW_{i}")

        return log_p
