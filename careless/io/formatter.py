import numpy as np
import tensorflow as tf
import reciprocalspaceship as rs
import gemmi
from .asu import ReciprocalASU,ReciprocalASUCollection
from careless.models.base import BaseModel
from careless.utils.positional_encoding import positional_encoding
from typing import Optional

def get_first_key_of_dtype(ds, dtype):
    matches = ds.dtypes[ds.dtypes == dtype].keys()
    for match in matches:
        return match
    return None

def standardize_metadata(metadata):
    """
    Standardize metadata ignoring columns with zero std deviation. 
    """
    std = metadata.std(0)
    zeros = (std == 0.)
    if np.any(zeros):
        import warnings
        warnings.warn("Metadata column with zero standard deviation will not be standardized.")

    metadata[:,~zeros] = (metadata[:,~zeros] - metadata[:,~zeros].mean(0)) / metadata[:,~zeros].std(0)
    return metadata
    

class DataFormatter():
    """
    Base class for formatting inputs. This class should not be used directly. To extend this class,
    implement a `prep_dataset` method which operates on a single dataset as well as a `finalize`
    method. `finalize(data, rac)` must accept a dataset and a reciprocal asu collection and return
    formatted model inputs 
    """
    spacegroups = None
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

    def prep_dataset(self, ds : rs.DataSet, spacegroup : Optional[gemmi.SpaceGroup] = None) -> rs.DataSet:
        raise NotImplementedError("Formatter classes should implement `prep_dataset`")

    def finalize(self, data : rs.DataSet, rac : ReciprocalASUCollection) -> (tuple, ReciprocalASUCollection):
        raise NotImplementedError("Formatter classes should implement `finalize`")

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
            sg = None
            if self.spacegroups is not None:
                sg = self.spacegroups[file_id]
            ds = self.prep_dataset(ds, sg)

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
                data = rs.concat((data, ds), check_isomorphous=False)

        if len(reciprocal_asus) == 0:
            reciprocal_asus.append(ReciprocalASU(data.cell, data.spacegroup, data.dHKL.min(), self.anomalous))

        rac = ReciprocalASUCollection(reciprocal_asus)
        data['image_id'] = data.groupby(['file_id', 'image_id']).ngroup()
        return data, rac

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
        data, rac = self.get_data_and_asu_collection(datasets)
        return self.finalize(data, rac)

    def format_files(self, files):
        """
        Parameters
        ----------
        files : iterable
            An iterable of filename strings.

        Returns
        -------
        inputs : tuple
            This is a nested tuple of numpy arrays formatted as merging inputs.
        asu_collection : careless.io.asu.ReciprocalASUCollection
            A collection of reciprocal asus to aid in intepreting results.
        """
        def load(filename):
            if filename.endswith('.mtz'):
                return rs.read_mtz(filename)
            elif filename.endswith('.stream'):
                return rs.read_crystfel(filename)

        return self((load(f) for f in files))

class MonoFormatter(DataFormatter):
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
            spacegroups = None,
            standardize = True,
        ):
        """
        TODO: fix this
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
        spacegroups : list (optional)
            Optional list of spacegroups. 1 Per input file
        standardize : bool (optional)
            Whether to standardize the metadata columns. The default is True.
        """
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
        self.spacegroups = spacegroups
        self.standardize = standardize

    @classmethod
    def from_parser(cls, parser):
        dmin = 0. if parser.dmin is None else parser.dmin
        pe_keys = parser.positional_encoding_keys
        if pe_keys is not None:
            pe_keys = pe_keys.split(',')

        spacegroups = None
        if parser.spacegroups is not None:
            spacegroups = [gemmi.SpaceGroup(i) for i in parser.spacegroups.split(",")]
            if len(spacegroups) == 1:
                spacegroups = [spacegroups[0] for i in range(len(parser.reflection_files))]
            elif len(spacegroups) != len(parser.reflection_files):
                raise ValueError(
                    "Multiple values provided for --spacegroups=, but the number of provided values does not match the number of reflection files. "
                    "Either provide a single spacegroup or one per reflection file as a comma-separated list. " 
                )

        return cls(
            parser.intensity_key,
            None, #<-- uncertainty key has to match {SIG,Sig}intensity_key
            parser.image_key,
            parser.metadata_keys.split(','),
            parser.separate_files,
            parser.anomalous,
            dmin,
            parser.isigi_cutoff,
            pe_keys,
            parser.positional_encoding_frequencies,
            spacegroups,
            standardize=parser.standardize_metadata,
        )

    def prep_dataset(self, ds, spacegroup=None, inplace=True):
        """
        Format a single data set.
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
        spacegroup : gemmi.SpaceGroup
            Optionally override ds.spacegroup with this object.
        inplace : bool (optional)
            By default this method operators inplace on the passed dataset.
            Set this parameter to False in order to operate on a copy.

        Returns
        -------
        standardized : rs.DataSet
        """
        if not inplace:
            ds = ds.copy()

        if spacegroup is not None:
            ds.spacegroup = spacegroup

        # Avoid non-unique MultiIndex complications
        ds.reset_index(inplace=True)

        # Resolution cutoff
        ds.compute_dHKL(inplace=True)
        ds.drop(ds.index[ds.dHKL < self.dmin], inplace=True)

        # Systematic absences
        ds.remove_absences(inplace=True)

        # Populate the observed miller indices before
        # switching the others into the ASU
        ds.loc[:,['Hobs', 'Kobs', 'Lobs']] = ds.get_hkls()

        # Map to ASU
        ds.hkl_to_asu(inplace=True, anomalous=self.anomalous)

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
        # TODO: this is a temporary fix for an older version of the crystfel parser
        # please remove this upon the next rs release (2021-09-10)
        if uncertainty_key is None:
            if 'sigmaI' in ds:
                uncertainty_key = 'sigmaI'

        if uncertainty_key is None:
            raise ValueError(
                f"No matching uncertainty key found for intensity key {intensity_key}"
                 "please manually specify the uncertainty key. "
            )

        # Add special IDs
        ds['intensity'] = ds[intensity_key]
        ds['uncertainty'] = ds[uncertainty_key]
        ds['image_id'] = ds[image_key]

        if self.isigi_cutoff is not None:
            ds.drop(ds.index[ds['intensity'] / ds['uncertainty'] < self.isigi_cutoff], inplace=True)

        return ds

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

        if self.standardize:
            metadata = standardize_metadata(metadata)

        if self.positional_encoding_keys is not None:
            to_encode = data[self.positional_encoding_keys].to_numpy('float32')
            encoded  = positional_encoding(to_encode, self.ecoding_bit_depth)
            metadata = np.concatenate((metadata, encoded), axis=1)

        refl_id = rac.to_refl_id(
            data['asu_id'].to_numpy('int64')[:,None],
            data.get_hkls(),
            )

        iobs    = data['intensity'].to_numpy('float32')[:,None]
        sigiobs = data['uncertainty'].to_numpy('float32')[:,None]

        inputs = {
            'refl_id'   : refl_id[:,None],
            'image_id'  : data['image_id'].to_numpy('int64')[:,None],
            'metadata'  : metadata,
            'intensities'   : iobs,
            'uncertainties' : sigiobs,
        }

        return self.pack_inputs(inputs), rac

class LaueFormatter(DataFormatter):
    def __init__(
            self, 
            wavelength_key,
            intensity_key,
            uncertainty_key,
            image_key,
            metadata_keys,
            separate_outputs,
            anomalous,
            lam_min=None,
            lam_max=None,
            dmin=0.,
            isigi_cutoff=None,
            positional_encoding_keys=None,
            encoding_bit_depth=5,
            spacegroups=None,
            standardize=True,
        ):

        """
        Parameters
        ----------
        anomalous : bool
            Whether to map to the anomalous ASU or proper ASU
        dmin : float (optional)
            The resolution cutoff in Å. By default no cutoff is applied.
        lam_min : float (optional)
            The minimum wavelength in Å. By default the lowest wavelength 
            in the "wavelength_key" column is used.
        lam_max : float (optional)
            The maximum wavelength in Å. By default the lowest wavelength 
            in the "wavelength_key" column is used.
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
        spacegroups : list (optional)
            Optional list of spacegroups. 1 Per input file.
        standardize : bool (optional)
            Whether to standardize the metadata columns. The default is True.
        """
        self.wavelength_key = wavelength_key
        self.lam_min = lam_min
        self.lam_max = lam_max
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
        self.spacegroups = spacegroups
        self.standardize = standardize

    @classmethod
    def from_parser(cls, parser):
        lmin = lmax = None
        if parser.wavelength_range is not None:
            lmin,lmax = parser.wavelength_range
        pe_keys = parser.positional_encoding_keys
        if pe_keys is not None:
            pe_keys = pe_keys.split(',')

        spacegroups = None
        if parser.spacegroups is not None:
            spacegroups = [gemmi.SpaceGroup(i) for i in parser.spacegroups.split(",")]
            if len(spacegroups) == 1:
                spacegroups = [spacegroups[0] for i in range(len(parser.reflection_files))]
            elif len(spacegroups) != len(parser.reflection_files):
                raise ValueError(
                    "Multiple values provided for --spacegroups=, but the number of provided values does not match the number of reflection files. "
                    "Either provide a single spacegroup or one per reflection file as a comma-separated list. " 
                )

        return cls(
            parser.wavelength_key,
            parser.intensity_key,
            None, #<-- uncertainty key has to match {SIG,Sig}intensity_key
            parser.image_key,
            parser.metadata_keys.split(','),
            parser.separate_files,
            parser.anomalous,
            lmin,
            lmax,
            parser.dmin,
            parser.isigi_cutoff,
            pe_keys,
            parser.positional_encoding_frequencies,
            spacegroups=None,
            standardize=parser.standardize_metadata,
        )

    def prep_dataset(self, ds, spacegroup=None):
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
        spacegroup : gemmi.SpaceGroup
            Optionally override ds.spacegroup with this object.

        Returns
        -------
        standardized : rs.DataSet
        """
        if spacegroup is not None:
            ds.spacegroup = spacegroup

        # Avoid non-unique MultiIndex complications
        ds.reset_index(inplace=True)

        # Populate the observed miller indices before
        # expanding harmonics
        ds.loc[:,['Hobs', 'Kobs', 'Lobs']] = ds.get_hkls()

        # Resolution cutoff
        ds.compute_dHKL(inplace=True) 
        dmin = self.dmin
        if dmin is None:
            dmin = ds.dHKL.min()
        ds.drop(ds.index[ds.dHKL < dmin], inplace=True)

        # Detect empirical wavelength range
        wavelength_key = self.wavelength_key
        lam_min = self.lam_min
        if lam_min is None:
            lam_min = ds[wavelength_key].min()
        lam_max = self.lam_max
        if lam_max is None:
            lam_max = ds[wavelength_key].max()

        # Expand all harmonics out to resolution cutoff
        from careless.utils.laue import expand_harmonics
        ds = expand_harmonics(ds, dmin, wavelength_key)

        # Filter by wavelength
        idx = (ds[wavelength_key] <= lam_min) | (ds[wavelength_key] >= lam_max)
        ds.drop(ds.index[idx], inplace=True)

        # Systematic absences
        ds.remove_absences(inplace=True)

        # Map to ASU
        ds.hkl_to_asu(inplace=True, anomalous=self.anomalous)

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
        # TODO: this is a temporary fix for an older version of the crystfel parser
        # please remove this upon the next rs release (2021-09-10)
        if uncertainty_key is None:
            if 'sigmaI' in ds:
                uncertainty_key = 'sigmaI'

        if uncertainty_key is None:
            raise ValueError(
                f"No matching uncertainty key found for intensity key {intensity_key}"
                 "please manually specify the uncertainty key. "
            )

        # Add special IDs
        ds['intensity'] = ds[intensity_key]
        ds['uncertainty'] = ds[uncertainty_key]
        ds['image_id'] = ds[image_key]

        isigi_cutoff = self.isigi_cutoff
        if isigi_cutoff is not None:
            ds.drop(ds.index[ds['intensity'] / ds['uncertainty'] < isigi_cutoff], inplace=True)

        return ds

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
        data = data.copy() #This is maybe overkill
        data['harmonic_id'] = data.groupby(['image_id', 'H_0', 'K_0', 'L_0']).ngroup()

        data['dHKL'] = data.dHKL**-2.
        metadata = data[self.metadata_keys].to_numpy('float32')

        if self.standardize:
            metadata = standardize_metadata(metadata)

        if self.positional_encoding_keys is not None:
            to_encode = data[self.positional_encoding_keys].to_numpy('float32')
            encoded  = positional_encoding(to_encode, self.ecoding_bit_depth)
            metadata = np.concatenate((metadata, encoded), axis=1)

        refl_id = rac.to_refl_id(
            data['asu_id'].to_numpy('int64')[:,None],
            data.get_hkls(),
            )

        iobs  = data[['harmonic_id',   'intensity']].groupby('harmonic_id').first().to_numpy('float32')
        sigma = data[['harmonic_id', 'uncertainty']].groupby('harmonic_id').first().to_numpy('float32')
        iobs  = np.pad( iobs, [[0, len(refl_id) - len( iobs)], [0, 0]], constant_values=1.)
        sigma = np.pad(sigma, [[0, len(refl_id) - len(sigma)], [0, 0]], constant_values=1.)

        inputs = {
            'refl_id'   : refl_id[:,None],
            'image_id'  : data['image_id'].to_numpy('int64')[:,None],
            'metadata'  : metadata,
            'intensities'   : iobs,
            'uncertainties'   : sigma,
            'wavelength' : data[self.wavelength_key].to_numpy('float32')[:,None],
            'harmonic_id' : data['harmonic_id'].to_numpy('int64')[:,None],
        }

        return self.pack_inputs(inputs), rac

