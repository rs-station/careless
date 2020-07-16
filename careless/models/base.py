import tensorflow as tf
import numpy as np


class BaseModel(object):
    """
    Base class for components in the postrefinement, scaling, and merging model. 
    Subclasses must implement __init__ and __call__.

    Attributes
    ----------
    trainable_variables : iterable
    """
    trainable_variables = []

    def __init__(self):
        e = NotImplementedError("Every model must implement __init__")
        raise e

#    def __call__(self):
#        e = NotImplementedError("Every model must implement __call__")
#        raise e


class PerGroupModel(BaseModel):
    """
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
        expanded = tf.sparse.sparse_dense_matmul(self.expansion_tensor, group_variables[:,None])[:,0]
        #expanded = tf.gather(group_variables, self.group_ids)
        return expanded

    def expand(self, *args):
        if self._use_gather:
            return self.expand_by_gather(*args)
        else:
            return self.expand_by_matmul(*args)

