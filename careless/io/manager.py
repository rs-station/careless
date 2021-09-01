import numpy as np
import tensorflow as tf
import reciprocalspaceship as rs
from .asu import ReciprocalASU,ReciprocalASUCollection
from careless.models.base import BaseModel
from careless.models.priors.wilson import WilsonPrior

class DataManager():
    """
    This class comprises various data manipulation methods as well as methods to aid in model construction.
    """
    def __init__(self, inputs, asu_collection):
        """
        Parameters
        ----------
        inputs : tuple
        asu_collection : ReciprocalASUCollection
        """
        self.inputs,self.asu_collection = inputs,asu_collection

    @classmethod
    def from_mtz_files(cls, filenames, formatter):
        return cls.from_datasets((rs.read_mtz(i) for i in filenames), formatter)

    @classmethod
    def from_stream_files(cls, filenames, formatter):
        return cls.from_datasets((rs.read_crystfel(i) for i in filenames), formatter)

    def get_wilson_prior(self, b=None):
        """ Construct a wilson prior with an optional temperature factor, b, appropriate for self.asu_collection. """
        if b is None:
            sigma = 1.
        elif isinstance(b, float):
            sigma = np.exp(-0.25 * b * self.asu_collection.dHKL**-2.)
        else:
            raise ValueError(f"parameter b has type{type(b)} but float was expected")

        return WilsonPrior(
            self.asu_collection.centric,
            self.asu_collection.multiplicity,
            sigma,
        )

    def get_tf_dataset(self, inputs=None):
        """
        Pack a dataset in the way that keras and careless expect.

        Parameters
        ----------
        inputs : tuple (optional)
            If None, self.inputs will be used
        """
        if inputs is None:
            inputs = self.inputs

        inputs = tuple(inputs)
        iobs = BaseModel.get_intensities(inputs)
        sigiobs = BaseModel.get_uncertainties(inputs)
        packed  = (inputs, iobs, sigiobs)
        tfds = tf.data.Dataset.from_tensor_slices(packed)
        return tfds.batch(len(iobs))

    def get_results(self, surrogate_posterior):
        """ 
        Extract results from a surrogate_posterior.

        Parameters
        ----------
        surrogate_posterior : tfd.Distribution
            A tensorflow_probability distribution or similar object with `mean` and `stddev` methods

        Returns
        -------
        results : tuple
            A tuple of rs.DataSet objects containing the results corresponding to each 
            ReciprocalASU contained in self.asu_collection
        """
        F = surrogate_posterior.mean().numpy()
        SigF = surrogate_posterior.stddev().numpy()
        asu_id,H = self.asu_collection.to_asu_id_and_miller_index(np.arange(len(F)))
        h,k,l = H.T
        refl_id = BaseModel.get_refl_id(self.inputs)
        N = np.bincount(refl_id.flatten(), minlength=len(F))
        results = ()
        for i,asu in enumerate(self.asu_collection):
            idx = asu_id == i
            output = rs.DataSet({
                'H' : h[idx],
                'K' : k[idx],
                'L' : l[idx],
                'F' : F[idx],
                'SigF' : F[idx],
                'N' : N[idx],
                }, 
                cell=asu.cell, 
                spacegroup=asu.spacegroup,
                merged=True,
            ).infer_mtz_dtypes().set_index(['H', 'K', 'L'])

            # Remove unobserved refls
            output = output[output.N > 0] 

            # Reformat anomalous data
            if asu.anomalous:
                output = output.unstack_anomalous()
                output = output[['F(+)', 'SigF(+)', 'F(-)', 'SigF(-)', 'N(+)', 'N(-)']]

            results += (output, )
        return results

    # <-- start xval data splitting methods
    def split_mono_data_by_mask(self, test_idx):
        """
        Method for splitting mono data given a boolean mask. 

        Parameters
        ----------
        test_idx : array (boolean)
            Boolean array with length of inputs.

        Returns
        -------
        train : tuple
        test  : tuple
        """
        test,train = (),()
        for inp in self.inputs:
            test  += (inp[ test_idx.flatten(),...] ,)
            train += (inp[~test_idx.flatten(),...] ,)
        return train, test

    def split_data_by_refl(self, test_fraction=0.5):
        """
        Method for splitting data given a boolean mask. 

        Parameters
        ----------
        test_fraction : float (optional)
            The fraction of reflections which will be reserved for testing.

        Returns
        -------
        train : tuple
        test  : tuple
        """
        if BaseModel.is_laue(self.inputs):
            harmonic_id = BaseModel.get_harmonic_id(self.inputs)
            test_idx = (np.random.random(harmonic_id.max()+1) <= test_fraction)[harmonic_id]
            train, test = self.split_laue_data_by_mask(test_idx)
            return self.get_tf_dataset(train), self.get_tf_dataset(test)

        test_idx = np.random.random(len(self.inputs[0])) <= test_fraction
        train, test = self.split_mono_data_by_mask(test_idx)
        return self.get_tf_dataset(train), self.get_tf_dataset(test)

    def split_laue_data_by_mask(self, test_idx):
        """
        Method for splitting laue data given a boolean mask. 
        This method will split up the data and alter the harmonic_id
        column to reflect the decrease in size of the array. 

        Parameters
        ----------
        test_idx : array (boolean)
            Boolean array with length of inputs.

        Returns
        -------
        train : tuple
        test  : tuple
        """
        harmonic_id = BaseModel.get_harmonic_id(self.inputs)

        # Let us just test that the boolean mask is valid for these data.
        # If it does not split observations, isect should be empty
        isect = np.intersect1d(
            harmonic_id[test_idx].flatten(),
            harmonic_id[~test_idx].flatten(),
        )
        if len(isect) > 0:
            raise ValueError(f"test_idx splits harmonic observations with harmonic_id : {isect}")

        def split(inputs, idx):
            harmonic_id = BaseModel.get_harmonic_id(inputs)

            result = ()
            uni,inv = np.unique(harmonic_id[idx], return_inverse=True)
            for i,v in enumerate(inputs):
                name = BaseModel.get_name_by_index(i)
                if name in ('intensities', 'uncertainties'):
                    v = v[uni]
                    v = np.pad(v, [[0, len(inv) - len(v)], [0, 0]])
                elif name == 'harmonic_id':
                    v = inv
                else:
                    v = v[idx.flatten(),...]
                result += (v ,)
            return result

        return split(self.inputs, ~test_idx), split(self.inputs, test_idx)

    def split_data_by_image(self, test_fraction=0.5):
        """
        Method for splitting data given a boolean mask. 
        This method will designate full images as belonging to the 
        train or test sets. 

        Parameters
        ----------
        test_fraction : float (optional)
            The fraction of images which will be reserved for testing.

        Returns
        -------
        train : tuple
        test  : tuple
        """
        image_id = BaseModel.get_image_id(self.inputs)
        test_idx = np.random.random(image_id.max()+1) <= test_fraction

        # Low image count edge case (mostly just for testing purposes)
        if True not in test_idx:
            test_idx[0] = True
        elif False not in test_idx:
            test_idx[0] = False
            
        test_idx = test_idx[image_id]
        if BaseModel.is_laue(self.inputs):
            train, test = self.split_laue_data_by_mask(test_idx)
        else:
            train, test = self.split_mono_data_by_mask(test_idx)

        return self.get_tf_dataset(train), self.get_tf_dataset(test)
    # --> end xval data splitting methods

