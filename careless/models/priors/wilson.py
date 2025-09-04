import reciprocalspaceship as rs
from careless.models.merging.surrogate_posteriors import RiceWoolfson
import tf_keras as tfk
from careless.utils.distributions import Stacy
from careless.models.priors.base import Prior
from tensorflow_probability import util as tfu
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
import tensorflow as tf
import numpy as np


class Centric(tfd.HalfNormal):
    def __init__(self, epsilon, sigma=1.):
        self.epsilon = tf.convert_to_tensor(epsilon)
        self.sigma = tf.convert_to_tensor(sigma)
        super().__init__(tf.math.sqrt(epsilon * self.sigma))


class Acentric(tfd.Weibull):
    def __init__(self, epsilon, sigma=1.):
        self.epsilon = tf.convert_to_tensor(epsilon)
        self.sigma = tf.convert_to_tensor(sigma)
        super().__init__(
            2., 
            tf.math.sqrt(self.epsilon * self.sigma),
        )

class WilsonPrior(Prior):
    """Wilson's priors on structure factor amplitudes."""
    def __init__(self, centric, epsilon, sigma=1.):
        """
        Parameters
        ----------
        centric : array
            Floating point or boolean array with value 1/True for centric reflections and 0/False. for acentric.
        epsilon : array
            Floating point array with multiplicity values for each structure factor.
        sigma : float or array
            The Σ value for the wilson distribution. The represents the average intensity stratified by a measure
            like resolution. 
        """
        super().__init__()
        self.epsilon = np.array(epsilon, dtype=np.float32)
        self.centric = np.array(centric, dtype=bool)
        self.sigma = np.array(sigma, dtype=np.float32)
        self.p_centric = Centric(self.epsilon, self.sigma)
        self.p_acentric = Acentric(self.epsilon, self.sigma)

    def log_prob(self, x):
        """
        Parameters
        ----------
        x : tf.Tensor
            Array of structure factor values with the same shape epsilon and centric.
        """
        return tf.where(self.centric, self.p_centric.log_prob(x), self.p_acentric.log_prob(x))

    def prob(self, x):
        """
        Parameters
        ----------
        x : tf.Tensor
            Array of structure factor values with the same shape epsilon and centric.
        """
        return tf.where(self.centric, self.p_centric.prob(x), self.p_acentric.prob(x))

    def mean(self):
        return tf.where(self.centric, self.p_centric.mean(), self.p_acentric.mean())

    def stddev(self):
        return tf.where(self.centric, self.p_centric.stddev(), self.p_acentric.stddev())

    def sample(self, *args, **kwargs):
        #### BLEERRRGG #####
        return tf.where(
            self.centric, 
            self.p_centric.sample(*args, **kwargs),
            self.p_acentric.sample(*args, **kwargs),
        )

class DoubleWilsonPrior(Prior):
    def __init__(self, asu_collection, parents, r_values, reindexing_ops=None, sigma=1., optimize_r=False):
        """
        asu_collection : AsuCollection
        parents : list
            List of integers such that parents[i] = j implies that asu_id==i has parent asu_id==j
        r_values : list or array
            Either a list with the same length as parents. 
        reindexing_ops : list or tuple
            A list of gemm.Op instances that is the same length as parents.
        sigma : float or array
            The scale parameter as in Wilson's priors
        optimize_r : bool (optional)
            Optionally allow r to optimize for non-root nodes. 
        """
        super().__init__()
        self.parents = parents
        self.optimize_r = optimize_r
        reflids = []
        loc = []
        scale = []
        root = []

        self.r = tf.convert_to_tensor(r_values, tf.float32)
        if optimize_r:
            self.r = tfu.TransformedVariable(
                self.r,
                tfb.Sigmoid(),
            )

        for child,parent in enumerate(parents):
            child_asu  = asu_collection.reciprocal_asus[child]
            child_size = len(child_asu.lookup_table)

            if parent is None:
                reflids.append(child_asu.lookup_table.id.to_numpy('int32'))
                root.append(np.ones(len(child_asu.lookup_table), dtype='bool'))
            else:
                root.append(np.zeros(len(child_asu.lookup_table), dtype='bool'))
                parent_asu = asu_collection.reciprocal_asus[parent]
                h = child_asu.Hall
                if reindexing_ops is not None:
                    op = reindexing_ops[child]
                    h = rs.utils.apply_to_hkl(h, op)
                h,_ = rs.utils.hkl_to_asu(h, parent_asu.spacegroup)
                pid = parent*np.ones((len(h), 1), dtype='int32')
                reflids.append(asu_collection.to_refl_id(pid, h, allow_missing=True))

        self.centric = asu_collection.centric
        self.multiplicity = asu_collection.multiplicity
        self.asu_ids = asu_collection.asu_ids

        self.sigma = sigma
        self.reflids = np.concatenate(reflids)
        self.absent = tf.convert_to_tensor(self.reflids == -1)
        self.root = np.concatenate(root)
        self.wilson_prior = WilsonPrior(asu_collection.centric, asu_collection.multiplicity, sigma)

    def mean(self):
        return self.wilson_prior.mean()

    def stddev(self):
        return self.wilson_prior.stddev()

    def log_prob(self, z):
        r = tf.gather(self.r, self.asu_ids)

        mask = self.reflids >= 0
        sanitized_reflids = tf.where(mask, self.reflids, 0)
        z_parent = tf.where(
            mask[None,:], 
            tf.gather(z, sanitized_reflids, axis=-1),
            0.,
        )

        loc = tf.where(
            self.absent, 
            0.,
            z_parent * r
        )
        r2 = tf.square(r)
        scale = tf.where(
            self.centric,
            tf.math.sqrt(self.multiplicity * self.sigma * (1. - r2)),
            tf.math.sqrt(0.5*self.multiplicity * self.sigma * (1. - r2)),
        )

        rice_woolfson = RiceWoolfson(loc, scale, self.centric)
        p_wilson = self.wilson_prior.log_prob(z)
        p_dw = rice_woolfson.log_prob(z)
        log_p = tf.where(self.root, p_wilson, p_dw)
        for i,r in enumerate(tf.unstack(self.r)):
            self.add_metric(r, f"rDW_{i}")
        return log_p

