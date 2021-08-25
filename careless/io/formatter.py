import numpy as np
import tensorflow as tf
import reciprocalspaceship as rs
from .asu import ReciprocalASU,ReciprocalASUCollection
from careless.models.base import BaseModel


def positional_encoding(X, L):
    """
    The positional encoding as defined in the NeRF paper https://arxiv.org/pdf/2003.08934.pdf
      gamma(p) = (sin(2**0*pi*p), cos(2**0*pi*p), ..., sin(2**(L-1)*pi*p), cos(2**(L-1)*pi*p))
    Wherein p represents an arbitrary batched set of vectors computed by normalizing X between
    between -1 and 1
    """
    p = 2.*(X - X.min(-2)) / (X.max(-2) - X.min(-2)) - 1.
    L = np.arange(L, dtype=X.dtype)
    f = np.pi*2**L
    fp = (f[...,None,:]*p[...,:,None]).reshape(p.shape[:-1] + (-1,))
    return np.concatenate((
        np.cos(fp),
        np.sin(fp),
    ), axis=-1)

def get_first_key_of_dtype(ds, dtype):
    matches = ds.dtypes[ds.dtypes == dtype].keys()
    for match in matches:
        return match
    return None

class BaseFormatter():
    def pack_inputs(self, inputs_dict):
        """
        inputs_dict : {k:v} where k corresponds to one of careless.models.base.BaseModel.input_index.keys()
        """
        inputs = ()
        for i in range(len(BaseModel.input_index)):
            k = BaseModel.get_name_by_index(i)
            if k in inputs_dict:
                inputs += (inputs_dict[k] , )
            else:
                break
        return inputs

    def prep_dataset(self, ds, anomalous, dmin=0., isigi_cutoff=None, image_key=None, intensity_key=None, uncertainty_key=None, inplace=True):
        """
        Standardized set of transformations to apply to input unmerged data sets.
         - Apply resolution cutoff (dHKL >= dmin)
         - Populate metadata 
            - "{H,K,L}obs"
            - "dHKL"
         - Map to ASU
         - Remove sys absences

        Parameters
        ----------
        ds : rs.DataSet
            The rs DataSet instance to be standardized
        anomalous : bool
            Whether to map to the anomalous ASU or proper ASU
        dmin : float (optional)
            The resolution cutoff in Å. By default no cutoff is applied.
        isigi_cutoff : float (optional)
            I/SigI cutoff
        image_key : string (optional)
            If this is not supplied, the first batch key will be used
        intensity_key : string (optional)
            If this is not supplied, the first intensity key will be used
        uncertainty_key : string (optional)
            If this is not supplied, the function will look for the intensity_key
            prepended by "Sig" or "SIG". If neither  exist, this method will
            raise a ValueError.
        inplace : bool (optional)
            By default this method operators inplace on the passed dataset.
            Set this parameter to False in order to operate on a copy.


        Returns
        -------
        standardized : rs.DataSet
        """
        if not inplace:
            ds = ds.copy()

        # Avoid non-unique MultiIndex complications
        ds.reset_index(inplace=True)

        # Resolution cutoff
        ds.compute_dHKL(inplace=True)
        ds.drop(ds.index[ds.dHKL < dmin], inplace=True)

        # Systematic absences
        ds.remove_absences(inplace=True)

        # Populate the observed miller indices before
        # switching the others into the ASU
        ds.loc[:,['Hobs', 'Kobs', 'Lobs']] = ds.get_hkls()

        # Map to ASU
        ds.hkl_to_asu(inplace=True, anomalous=anomalous)

        # Try to guess the image key
        image_key = self.image_key
        if image_key is None:
            image_key = get_first_key_of_dtype(ds, 'B')

        # Try to guess the intensity key
        intensity_key = self.intensity_key
        if intensity_key is None:
            intensity_key = get_first_key_of_dtype(ds, 'J')

        # Try to guess the uncertainty key
        uncertainty_key = self.uncertainty_key
        if uncertainty_key is None:
            for prefix in ['Sig', 'SIG']:
                for k in ds:
                    if k == prefix + intensity_key:
                        uncertainty_key = k
        if uncertainty_key is None:
            raise ValueError(
                f"No matching uncertainty key found for intensity key {intensity_key}"
                 "please manually specify the uncertainty key. "
            )

        # Add special IDs
        ds['intensity'] = ds[intensity_key]
        ds['uncertainty'] = ds[uncertainty_key]
        ds['image_id'] = ds[image_key]

        if isigi_cutoff is not None:
            ds.drop(ds.index[ds['intensity'] / ds['uncertainty'] < isigi_cutoff], inplace=True)

        return ds

class MonoFormatter(BaseFormatter):
    """
    Formatter for careless inputs. 
    """
    def __init__(
            self, 
            intensity_key,
            uncertainty_key,
            image_key,
            metadata_keys,
            separate_outputs,
            anomalous,
            dmin=0.,
            isigi_cutoff=None,
            positional_encoding_keys=None,
            encoding_bit_depth=5,
        ):
        self.intensity_key = intensity_key
        self.uncertainty_key = uncertainty_key
        self.image_key = image_key
        self.metadata_keys = metadata_keys
        self.positional_encoding_keys = positional_encoding_keys
        self.separate_outputs = separate_outputs
        self.anomalous = anomalous
        self.dmin = dmin
        self.isigi_cutoff = isigi_cutoff
        self.positional_encoding_keys = positional_encoding_keys
        self.ecoding_bit_depth = encoding_bit_depth

    @classmethod
    def from_argparse(cls, parser):
        raise NotImplementedError()

    def get_data_and_asu_collection(self, datasets):
        """
        Parameters
        ----------
        datasets : iterable
            An iterable of rs.DataSet objects

        Returns
        -------
        data : rs.DataSet
            This is an annotated dataset concatenating all the datasets.
        asu_collection : careless.io.asu.ReciprocalASUCollection
            A collection of reciprocal asus to aid in intepreting results.
        """
        data = None

        reciprocal_asus = []
        for file_id, ds in enumerate(datasets):
            ds = self.prep_dataset(
                ds, 
                self.anomalous, 
                self.dmin, 
                image_key = self.image_key,
                intensity_key = self.intensity_key,
                uncertainty_key = self.uncertainty_key,
                inplace=True,
            )

            if self.separate_outputs:
                asu_id = file_id
                reciprocal_asus.append(ReciprocalASU(ds.cell, ds.spacegroup, ds.dHKL.min(), self.anomalous))
            else:
                asu_id = 0

            ds['file_id']  = file_id
            ds['asu_id']   = asu_id

            if data is None:
                data = ds
            else:
                data = data.append(ds)

        if len(reciprocal_asus) == 0:
            reciprocal_asus.append(ReciprocalASU(data.cell, data.spacegroup, data.dHKL.min(), self.anomalous))

        rac = ReciprocalASUCollection(reciprocal_asus)
        data['image_id'] = data.groupby(['file_id', 'image_id']).ngroup()
        return data, rac

    def finalize(self, data, rac):
        """
        Parameters
        ----------
        data : rs.DataSet
            This is an annotated dataset concatenating all the datasets.
        asu_collection : careless.io.asu.ReciprocalASUCollection
            A collection of reciprocal asus to aid in intepreting results.

        Returns
        -------
        inputs : tuple
            This is a tuple of numpy arrays formatted as merging inputs.
        asu_collection : careless.io.asu.ReciprocalASUCollection
            A collection of reciprocal asus to aid in intepreting results.
        """
        data['dHKL'] = data.dHKL**-2.
        metadata = data[self.metadata_keys].to_numpy('float32')
        metadata = (metadata - metadata.mean(0)) / metadata.std(0)

        if self.positional_encoding_keys is not None:
            to_encode = data[self.positional_encoding_keys].to_numpy('float32')
            encoded  = positional_encoding(to_encode, self.ecoding_bit_depth)
            metadata = np.concatenate((metadata, encoded), axis=1)

        refl_id = rac.to_refl_id(
            data['asu_id'].to_numpy('int64')[:,None],
            data.get_hkls(),
            )

        inputs = {
            'refl_id'   : refl_id[:,None],
            'image_id'  : data['image_id'].to_numpy('int64')[:,None],
            'metadata'  : metadata,
            'intensities'   : data['intensity'].to_numpy('float32')[:,None],
            'uncertainties' : data['uncertainty'].to_numpy('float32')[:,None],
        }

        return self.pack_inputs(inputs), rac

    def __call__(self, datasets):
        """
        Parameters
        ----------
        datasets : iterable
            An iterable of rs.DataSet objects

        Returns
        -------
        inputs : tuple
            This is a nested tuple of numpy arrays formatted as merging inputs.
        asu_collection : careless.io.asu.ReciprocalASUCollection
            A collection of reciprocal asus to aid in intepreting results.
        """
        image_id_offset = 0
        data = None

        data, rac = self.get_data_and_asu_collection(datasets)
        return self.finalize(data, rac)

class LaueFormatter(MonoFormatter):
    def __init__(self, *args, wavelength_key='Wavelength', , **kwargs):
        pass

    def prep_dataset(self, *args, **kwargs):
        from careless.utils.laue import expand_harmonics
        ds = super().prep_dataset(*args, **kwargs)
        expand_harmonics(ds, dmin, wavelength_key)
