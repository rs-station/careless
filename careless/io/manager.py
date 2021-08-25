import numpy as np
import tensorflow as tf
import reciprocalspaceship as rs
from .asu import ReciprocalASU,ReciprocalASUCollection
from careless.models.base import BaseModel
from careless.models.priors.wilson import WilsonPrior


class Manager():
    """
    This class comprises various data manipulation methods as well as methods to aid in model construction.
    """
    inputs
    asu_collection 

    def __init__(self, datasets, formatter):
        """
        Parameters
        ----------
        datasets : iterable
            A list or other iterable of rs.DataSet objects.
        formatter : careless.io.loader.DataFormatter
            A DataFormatter instance or other similar callable which takes a sequence of
            datasets and returns a tuple, (inputs : (x, y, sigy), asu_collection : ReciprocalASUCollection)
        """
        self.formatter = formatter
        self.inputs,self.asu_collection = formatter(datasets)

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
        return tfds.batch(iobs)

    def split_laue_data_by_refl(self, test_fraction=0.5):
        """
        Returns
        -------
        train : tf.data.DataSet
        test  : tf.data.DataSet
        """
        harmonic_id = BaseModel.get_harmonic_id(self.inputs)
        test_idx = (np.random.random(harmonic_id.max()) <= test_fraction)[harmonic_id]

        train = []
        test  = []

        def reformat_inputs(inputs, idx):
            harmonic_id = BaseModel.get_harmonic_id(inputs)
            intensity_idx = np.sort(np.unique(harmonic_id[idx]))

            data = []
            for i,v in enumerate(inputs):
                name = BaseModel.get_name_by_index(i)
                if name in ("intensities", "uncertainties"):
                    data.append(v[intensity_idx])
                elif name == 'harmonic_id':
                    _,v = np.unique(v[idx], return_inverse=True)
                    data.append(v)
                else:
                    data.append(v[idx])
            return data

        train = self.get_tf_dataset(reformat_inputs(inputs, ~idx))
        test  = self.get_tf_dataset(reformat_inputs(inputs, idx))
        return train,test

    def split_data_by_refl(self, test_fraction=0.5):
        """
        Returns
        -------
        train : tf.data.DataSet
        test  : tf.data.DataSet
        """

        if len(self.inputs) >= BaseModel.input_index['harmonic_id']:
            return self.split_laue_data_by_refl(self, test_fraction)

        refl_id = BaseModel.get_refl_id(self.inputs)
        idx = np.random.random(len(refl_id)) <= test_fraction

        train = self.get_tf_dataset([i[~idx] for i in self.inputs])
        test  = self.get_tf_dataset([ i[idx] for i in self.inputs])

        return train,test

    def split_data_by_image(self):
        """
        Returns
        -------
        train : tf.data.DataSet
        test  : tf.data.DataSet
        """
        image_id = BaseModel.get_image_id(even_data.as_numpy_iterator().next()[0])
        idx = (np.random.random(image_id.max()+1) <= test_fraction)[image_id]

        train = self.get_tf_dataset([i[~idx] for i in self.inputs])
        test  = self.get_tf_dataset([ i[idx] for i in self.inputs])

        return train,test

    def get_results(self, model):
        """ Extract results from a VariationalMergingModel instance """
        F = model.surrogate_posterior.mean().numpy()
        SigF = model.surrogate_posterior.stddev().numpy()
        asu_id,H = self.asu_collection.to_asu_id_and_miller_index(np.arange(len(F)))
        h,k,l = H.T
        refl_id = BaseModel.get_refl_id(even_data.as_numpy_iterator().next()[0])
        N = np.bincount(refl_id, minlenght=len(F))
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

            yield output

