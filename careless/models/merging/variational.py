from careless.models.base import PerGroupModel
from careless.utils.shame import sanitize_tensor
from careless.models.distributions.normalized import Acentric,Centric
from careless.utils.math import softplus_inverse
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np

class VariationalMergingModel(PerGroupModel):
    """
    Merge data with a posterior parameterized by a truncated normal distribution.
    """
    def __init__(self, iobs, sig_iobs, miller_ids, epsilons, centric, corrections, t_dof=None, posterior_truncated_normal_max=100000.):
        """"
        Parameters
        ----------
        iobs : array                                                                                                
            array of observed reflection intensities                                                                
        sig_iobs : array                                                                                            
            error estimates for observed reflection intensities from integration                                    
        miller_ids : array(int)
            array of zero indexed integers with an id corresponding to each unique miller index.
        centric : array(float)                                                                                        
            array with `length == miller_ids.max() + 1`  which has ones for centric reflections and zeros elsewhere
        epsilons : array(float)                                                                                        
            array with `length == miller_ids.max() + 1` which has the multiplicity corrections for each reflection
        corrections : list
            list of correction models to iobs
        t_dof : float (optional)
            If supplied use a t-distributed error model with these degrees of freedom. The default is None => normal loss.
        posterior_truncated_normal_max : float
            the maximum value of the surrogate posterior distribution. this defaults to 100. which is sufficient for normalized structure factor amplitudes
        """
        super().__init__(miller_ids)
        self.corrections = corrections
        self.posterior_truncated_normal_max = posterior_truncated_normal_max

        self.centric = np.array(centric, dtype=np.float32)
        self.epsilons = np.array(epsilons, dtype=np.float32)

        centric_loc = 5.
        #centric_scale = softplus_inverse(2.0)
        centric_scale = 2.

        acentric_loc = 5.
        #acentric_scale = softplus_inverse(2.0)
        acentric_scale = 2.

        self._q_loc_init = self.centric*centric_loc + (1. - self.centric)*acentric_loc
        self._q_scale_init = self.centric*centric_scale + (1. - self.centric)*acentric_scale

        self.posterior_loc = tf.Variable(self._q_loc_init)
        self._posterior_scale = tf.Variable(self._q_scale_init)

        self.trainable_variables = tf.nest.flatten([
            self.posterior_loc,
            self._posterior_scale,
        ] + [c.trainable_variables for c in self.corrections])

        self.sig_iobs = sig_iobs
        self.iobs = iobs
        self.dof = t_dof

    @property
    def posterior_scale(self):
        return tf.math.softplus(self._posterior_scale)
        #return tf.nn.relu(self._posterior_scale)

    def get_corrections(self):
        corrections = 1.
        for c in self.corrections:
            corrections *= c()
        return corrections

    @property
    def variational_distribution(self):
        loc = self.posterior_loc
        scale  = self.posterior_scale
        q = tfp.distributions.TruncatedNormal(loc=loc, scale=scale, low=0., high=self.posterior_truncated_normal_max)
        return q

    def predict(self):
        q = self.variational_distribution
        E = q.mean()
        corrections = self.get_corrections()
        I = self.expand(self.epsilons * E**2.) * corrections
        return I

    def get_error_distributions(self):
        if self.dof is None:
            dists = tfp.distributions.Normal(self.iobs, self.sig_iobs)
            return dists
        elif self.dof < 0.:
            dists = tfp.distributions.Laplace(self.iobs, np.sqrt(2.)*self.sig_iobs)
            return dists
        else:
            dists = tfp.distributions.StudentT(self.dof, self.iobs, self.sig_iobs)
            return dists

    @property
    def normalized_structure_factors(self):
        return self.variational_distribution.mean()

    @property
    def normalized_structure_factor_errors(self):
        return self.variational_distribution.stddev()

    def __call__(self):
        """
        Returns
        -------
        loss : Tensor
            The scalar value of the Evidence Lower BOund.
        """
        #Variational distributions
        q = self.variational_distribution

        #Log likelihood
        E = q.sample()
        I = self.epsilons*E**2
        corrections = self.get_corrections()
        dists = self.get_error_distributions()
        log_probs = dists.log_prob(self.expand(I)*corrections)
        log_likelihood = tf.reduce_sum(log_probs)

        #Prior Distribution
        p_centric = Centric()
        p_acentric = Acentric()

        #Prior probs
        p_E = p_centric.prob(E)*self.centric + p_acentric.prob(E)*(1. - self.centric)
        q_E = q.prob(E/100.)

        eps = 1e-12
        kl_div = tf.reduce_sum(q_E * (tf.math.log(q_E + eps) - tf.math.log(p_E + eps)))
        loss = -log_likelihood + kl_div
        return loss

    def loss_and_grads(self, variables, s=1):
        with tf.GradientTape() as tape:
            loss = 0.
            for i in range(s):
                loss += self()/s
        grads = tape.gradient(loss, variables)
        return loss, grads

    def is_valid_gradient(self, grads):
        invalid = tf.reduce_any([tf.reduce_any(~tf.math.is_finite(i)) for i in grads])
        return ~invalid

    @tf.function
    def train_step(self, optimizer, s=1):
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

        variables = self.trainable_variables
        loss, grads = self.loss_and_grads(variables, s)
        grads = [sanitize_tensor(g) for g in grads]
        optimizer.apply_gradients(zip(grads, variables))
        return loss

    def rescue_variational_distributions(self):
        """
        Reset values for problem distributions to their intial values. 
        """
        q = self.variational_distribution
        q_E = q.prob(q.sample())
        self.posterior_loc.assign(
            tf.where(tf.math.is_finite(q_E), self.posterior_loc, self._q_loc_init)
        )
        self._posterior_scale.assign(
            tf.where(tf.math.is_finite(q_E), self.posterior_scale, self._q_scale_init)
        )
        

class VariationalHarmonicMergingModel(VariationalMergingModel):
    """
    Merging model for polychromatic X-ray diffraction data which may contain harmonically overlapped spots. 
    This class works like other variational models in this package by maximizing the evidence lower bound between
    a truncated normal posterior surrogate distribution and the historical Wilson priors on normalized
    structure factor amplitudes.

    The difference between this class and others is that all reflections are predicted, corrected, and subsequently
    harmonics are summed to square with the observed intensities. Briefly, the algorithm works as follows

    1. Samples are drawn from the surrogate posterior
        - at this point, there is one sampled structure factor amplitude per unique miller index in the reciprocal asu
    2. Samples are corrected for reflection multiplicity
        - this refers to the epsilon factors which are a spacegroup dependent correction for reciprocal space symmetry
    3. Samples are expanded to the size of the full set of harmonically deconvolved reflection observations and squared
        - at this stage there will be __more__ reflections than were actually observed in the data set. 
    4. Expanded samples are now corrected by whatever correction objects were passed to this class's constructor
    5. Harmonic samples are merged 
        - now the samples will have the same number of entries as the observed data set.
    6. The log likelihood and kl div are now computed with respect to the harmonically convolved samples
    """
    def __init__(self, iobs, sig_iobs, miller_ids, epsilons, centric, corrections, harmonic_index, t_dof=None):
        """"
        Constructing this object is a little more complex than a monochromatic merging model. 
        In particular, the user needs to pay attention to the lengths of the arrays being passed to the constructor. 
        In general, 
        
        >>> len(epsilons) == len(centric) <= len(iobs) == len(sig_iobs) <= len(miller_ids) == len(harmonic_index)

        Parameters
        ----------
        iobs : array                                                                                                
            array of observed reflection intensities. the length of this is the actual number of integrated refl intensities.
        sig_iobs : array                                                                                            
            error estimates for observed reflection intensities from integration. the length of this is the actual number of integrated refl intensities.
        miller_ids : array(int)                                                                                        
            zero indexed array of integer reflection indices in the length of the full set of haromically convolved reflections. For a typical laue data set, this should be longer than iobs.
        centric : array(float)                                                                                        
            array with `length == miller_ids.max() + 1`  which has ones for centric reflections and zeros elsewhere
        epsilons : array(float)                                                                                        
            array with `length == miller_ids.max() + 1` which has the multiplicity corrections for each reflection
        corrections : list
            list of correction models
        harmonic_index : array(int)
            the harmonic index groups reflections from the same image and central ray. it should have the same length as miller_ids and harmonic_index.max() + 1 is the number of reflection observations. simulated reflections with harmonic index n will be summed to predict the iobs[n].
        t_dof : float (optional)
            If supplied use a t-distributed error model with these degrees of freedom. The default is None => normal loss.
        """
        super().__init__(iobs, sig_iobs, miller_ids, epsilons, centric, corrections, t_dof=None)
        # The base class takes care of most of the setup for us. However, we need a sparse tensor for merging
        # Harmonically convolved reflections. The PerGroupModel constructure will take care of this for us.
        # Formally, PerGroupModel(harmonic_index).expansion_tensor is the transpose of what we need.  
        self.harmonic_index = harmonic_index
        self.harmonic_convolution_tensor = PerGroupModel(harmonic_index).expansion_tensor

    def convolve(self, tensor):
        """
        Paramters
        ---------
        tensor : tf.Tensor
            array of predicted reflection intensities with length self.harmonic_convolution_tensor.shape[1]
        
        Returns
        -------
        convolved : tf.Tensor
            array of predicted reflection intensities which have been convolved by a sparse matmul
        """
        convolved = tf.squeeze(tf.sparse.sparse_dense_matmul(
            self.harmonic_convolution_tensor, 
            tf.expand_dims(tensor, -1), 
            adjoint_a=True
        ))
        return convolved

    @tf.function
    def __call__(self):
        """
        Returns
        -------
        loss : Tensor
            The scalar value of the Evidence Lower BOund.
        """
        #Variational distributions
        q = self.variational_distribution

        #Log likelihood
        E = q.sample()
        I = self.epsilons*E**2
        corrections = self.get_corrections()
        dists = self.get_error_distributions()
        I_raw = self.expand(I)*corrections
        I = self.convolve(I_raw) #This is the only thing changed relative to the base VariationalMergingModel
        log_probs = dists.log_prob(I)
        log_likelihood = tf.reduce_sum(log_probs)

        #Prior Distribution
        p_centric = Centric()
        p_acentric = Acentric()

        #Prior probs
        p_E = p_centric.prob(E)*self.centric + p_acentric.prob(E)*(1. - self.centric)
        q_E = q.prob(E/100.)

        eps = 1e-12
        kl_div = tf.reduce_sum(q_E * (tf.math.log(q_E + eps) - tf.math.log(p_E + eps)))
        for c in self.corrections:
            if hasattr(c, 'kl_term'):
                kl_div += c.kl_term()

        loss = -log_likelihood + kl_div
        return loss

class WeightedVariationalHarmonicMergingModel(VariationalHarmonicMergingModel):
    @tf.function
    def __call__(self):
        """
        Returns
        -------
        loss : Tensor
            The scalar value of the Evidence Lower BOund.
        """
        #Variational distributions
        q = self.variational_distribution

        #Log likelihood
        E = q.sample()
        I = self.epsilons*E**2
        corrections = self.get_corrections()
        dists = self.get_error_distributions()
        I_raw = self.expand(I)*corrections
        I = self.convolve(I_raw) #This is the only thing changed relative to the base VariationalMergingModel
        log_probs = dists.log_prob(I)
        #raise NotImplementedError("Look into different functional forms for weights!")
        weights = self.iobs / self.sig_iobs
        log_probs = log_probs*weights
        log_likelihood = tf.reduce_sum(log_probs)

        #Prior Distribution
        p_centric = Centric()
        p_acentric = Acentric()

        #Prior probs
        p_E = p_centric.prob(E)*self.centric + p_acentric.prob(E)*(1. - self.centric)
        q_E = q.prob(E/100.)

        eps = 1e-12
        kl_div = tf.reduce_sum(q_E * (tf.math.log(q_E + eps) - tf.math.log(p_E + eps)))
        for c in self.corrections:
            if hasattr(c, 'kl_term'):
                kl_div += c.kl_term()

        loss = -log_likelihood + kl_div
        return loss
