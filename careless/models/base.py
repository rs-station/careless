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
        'file_id'       : 2,
        'metadata'      : 3,
        'intensities'   : 4,
        'uncertainties' : 5,
        'wavelength'    : 6,
        'harmonic_id'   : 7,
    }

    def call(self, inputs):
        raise NotImplementedError(
            "All Scaler classes must implement a call method which accepts inputs "
            "defined by this class.                                               "
        )

    @staticmethod
    def is_laue(inputs : tuple) -> bool:
        """
        Test if the inputs are from Laue or Mono data
        """
        laue_size = BaseModel.get_index_by_name("harmonic_id") + 1
        if len(inputs) >= laue_size:
            return True
        return False

    @staticmethod
    def get_name_by_index(index : int) -> str:
        for k,v in BaseModel.input_index.items():
            if v == index:
                return k
        raise ValueError(
            f"index, {index}, not a valid index. Valid indices are {BaseModel.input_index.values()}."
        )

    @staticmethod
    def get_index_by_name(name):
        if name not in BaseModel.input_index:
            raise ValueError(
                f"name, {name}, not a valid key. Valid keys are {BaseModel.input_index.keys()}."
            )
        return BaseModel.input_index[name]

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
        if datum.shape[0] == 1:
            datum = tf.squeeze(datum, axis=0)
        return datum

    @staticmethod
    def get_refl_id(inputs):
        """ Given a collection of inputs extract just the reflection_id """
        return BaseModel.get_input_by_name(inputs, 'refl_id')

    @staticmethod
    def get_file_id(inputs):
        """ Given a collection of inputs extract just the file_id """
        return BaseModel.get_input_by_name(inputs, 'file_id')

    @staticmethod
    def get_image_id(inputs):
        """ Given a collection of inputs extract just the image_id """
        return BaseModel.get_input_by_name(inputs, 'image_id')

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

