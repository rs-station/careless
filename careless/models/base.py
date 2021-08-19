import tensorflow as tf
from tensorflow import keras as tfk
import numpy as np



class BaseModel(tfk.layers.Layer):
    """ 
    Base class for all models in `careless`. 
    It encodes accessors for the standard format inputs to the model. 
    When extending this class, use the get_{
        metadata,
        refl_id,
        image_id,
        intensities,
        uncertainties,
        wavelength,
        harmonic_id
    } static methods to parse your inputs. This ensures it is easy to
    change the data format in the future.
    """
    input_index = {
        'refl_id'       : 0,
        'image_id'      : 1,
        'metadata'      : 2,
        'intensities'   : 3,
        'uncertainties' : 4,
        'wavelength'    : 5,
        'harmonic_id'   : 6,
    }

    def call(self, inputs):
        raise NotImplementedError(
            "All Scaler classes must implement a call method which accepts inputs "
            "which are a sequence inputs = [refl_id, image_id, metadata] and      "
            "returns a tfp.distributions.Distribution.                            "
        )

    @staticmethod
    def get_input_by_name(inputs, name):
        if name not in BaseModel.input_index:
            raise ValueError(
                f"name, {name}, not a valid key. Valid keys are {BaseModel.input_index.keys()}."
            )
        idx = BaseModel.input_index[name]
        try:
            datum = inputs[idx]
        except:
            raise ValueError(
                f"Attempting to gather {name} data from input tensors, {inputs}, with length {len(inputs)} failed."
            )
        return datum

    @staticmethod
    def get_refl_id(inputs):
        """ Given a collection of inputs extract just the reflection_id """
        return BaseModel.get_input_by_name(inputs, 'refl_id')

    @staticmethod
    def get_image_id(inputs):
        """ Given a collection of inputs extract just the image_id """
        return BaseModel.get_input_by_name(inputs, 'input_id')

    @staticmethod
    def get_metadata(inputs):
        """ Given a collection of inputs extract just the metadata """
        return BaseModel.get_input_by_name(inputs, 'metadata')

    @staticmethod
    def get_intensities(inputs):
        """ Given a collection of inputs extract just the intensities """
        return BaseModel.get_input_by_name(inputs, 'intensities')

    @staticmethod
    def get_uncertainties(inputs):
        """ Given a collection of inputs extract just the uncertainty estimates """
        return BaseModel.get_input_by_name(inputs, 'uncertainties')

    @staticmethod
    def get_wavelength(inputs):
        """ Given a collection of inputs extract just the wavelength. This method only applies to Laue data."""
        return BaseModel.get_input_by_name(inputs, 'wavelength')

    @staticmethod
    def get_harmonic_id(inputs):
        """ Given a collection of inputs extract just the harmonic_id. This method only applies to Laue data."""
        return BaseModel.get_input_by_name(inputs, 'harmonic_id')

class PerXXGroupModel():
    """
    This legacy class is no longer used, but it contains some useful snippets regarding sparse tensor ops. 

    Base class for corrections that are applied to reflection observations by some grouping.
    This uses tensorflow SparseTensor multiplication to take care of indexing.

    Attributes
    ----------
    group_ids : array(int)
        Zero indexed array of integer indices indicating which group each reflection observation belongs to. 
    expansion_tensor : tf.SparseTensor
        Sparse tensor with expansion_tensor.shape == (number of reflection obs,  number of groups). This can 
        be used to distribute variables to groups of observations based on observation_ids.
    num_observations : int
        Number of reflection observations
    num_groups : int
        Number of groups into which the reflection observations are subdivided
    """
    group_ids = None
    expansion_tensor = None
    num_observations = None
    num_groups = None

    def __init__(self, group_ids):
        """
        Parameters
        ----------
        group_ids : array(int)
            Zero indexed array of integer indices indicating which group each reflection observation belongs to. 
        """
        #self.group_ids = tf.convert_to_tensor(group_ids, dtype=tf.int64)
        self.group_ids = np.array(group_ids, dtype=np.int64)
        self.num_groups = int(max(group_ids)) + 1
        self.num_observations = group_ids.shape[0]
        self._use_gather = False

        idx = tf.concat((tf.range(self.num_observations, dtype=tf.int64)[:,None], self.group_ids[:,None]), 1) 
        shape = [self.num_observations, self.num_groups]
        self.expansion_tensor = tf.SparseTensor(idx, tf.ones(self.num_observations, dtype=tf.float32), shape)

    def expand_by_gather(self, group_variables):
        """
        Expand a tensor in the dimension of the number of groups to the dimension of reflection observations. 
        This is just a tf.gather operation under the hood. This implementation is much slower for optimization,
        but it place nicer when trying to compute higher order gradients. 

        Parameters
        ----------
        group_variables : tf.Tensor
            1D tensor of variables with length num_groups

        Returns
        -------
        expanded : tf.Tensor
            1D tensor with length num_observations
        """
        expanded = tf.gather(group_variables, self.group_ids)
        return expanded

    def expand_by_matmul(self, group_variables):
        """
        Expand a tensor in the dimension of the number of groups to the dimension of reflection observations. 
        This is just a sparse tensor matmul under the hood. 

        Parameters
        ----------
        group_variables : tf.Tensor
            1D tensor of variables with length num_groups

        Returns
        -------
        expanded : tf.Tensor
            1D tensor with length num_observations
        """
        if len(group_variables.shape) == 1:
            expanded = tf.sparse.sparse_dense_matmul(self.expansion_tensor, group_variables[:,None])[:,0]
        else:
            expanded = tf.transpose(tf.sparse.sparse_dense_matmul(self.expansion_tensor, group_variables, adjoint_b=True))
        #expanded = tf.gather(group_variables, self.group_ids)
        return expanded

    def expand(self, *args):
        if self._use_gather:
            return self.expand_by_gather(*args)
        else:
            return self.expand_by_matmul(*args)

