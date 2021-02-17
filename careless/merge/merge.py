import gemmi
import numpy as np
import tensorflow as tf
import pandas as pd
import reciprocalspaceship as rs
from careless.models.merging.variational import VariationalMergingModel

#pd.options.mode.chained_assignment = 'raise'


def get_first_key_of_type(ds, typestring):
    idx = ds.dtypes==typestring
    if idx.sum() < 1:
        #raise KeyError(f"No key matching typestring {typestring}")
        return None
    else:
        return ds.dtypes[idx].keys()[0]
        

class BaseMerger():
    """
    This Merger object is for combining all experiments into a single set of Structure Factors.

    Special keys in MergerBase.data
        file_id   : numeric index for mtz file populated at load time
        miller_id : unique numeric index of miller indices
        observation_id : unique index for each reflection observation

    Attributes
    ----------
    results : rs.DataSet
    data : rs.DataSet
    merger : careless.models.merging.VariationalMergingModel
    """
    results = None
    merger = None
    data = None
    spacegroups = None
    cells = None
    prior = None
    likelihood = None
    scaling_model = None
    intensity_key = None
    sigma_intensity_key = None
    anomalous = False
    surrogate_posterior = None

    def __init__(self, datasets, anomalous=False, dmin=None, isigi_cutoff=None, intensity_key=None, weight_kl=False):
        """
        Parameters
        ----------
        datasets : iterable
            An iterable containing rs.DataSet object(s).
        anomalous : bool
            True => don't merge friedel mates
        dmin : float
            Max resolution.
        isigi_cutoff : float
            Minimum I/Sigma to keep. 
        intensity_key : str
            The name of the column containing  the intensities to be merged. There must be a corresponding 
            standard deviation key with name 'Sig' + intensity_key or 'SIG' + intensity_key
        """
        self.data = None
        self.cells = []
        self.spacegroups = []
        self.weight_kl = weight_kl
        for i,ds in enumerate(datasets):
            ds = ds.copy() #Out of an abundance of caution
            ds.reset_index(inplace=True)

            ds[['Hobs', 'Kobs', 'Lobs']] = ds.loc[:,['H', 'K', 'L']]
            ds.hkl_to_asu(inplace=True)

            if anomalous:
                self.anomalous = True
                friedel_sign = np.array([-1., 1.])[ds['M/ISYM'] % 2]
                friedel_sign[ds.label_centrics().CENTRIC] = 1.
                ds.loc[:,['H', 'K', 'L']] = friedel_sign[:,None] * ds.loc[:,['H', 'K', 'L']]
                ds['FRIEDEL'] = friedel_sign

            self.spacegroups.append(ds.spacegroup)
            self.cells.append(ds.cell)
            ds['file_id'] = i
            self.data = ds.append(self.data, check_isomorphous=False)

        self.data.cell = self.data.spacegroup = None

        # Try to guess sensible default keys. 
        # The user can change after the constructor is finished
        self.intensity_key = intensity_key
        if self.intensity_key is None:
            self.intensity_key = get_first_key_of_type(self.data, "J")
        if f'Sig{self.intensity_key}' in self.data:
            self.sigma_intensity_key = f'Sig{self.intensity_key}'
        elif f'SIG{self.intensity_key}' in self.data:
            self.sigma_intensity_key = f'SIG{self.intensity_key}'
        else:
            self.sigma_intensity_key = get_first_key_of_type(self.data, "Q")

        self.metadata_keys = list(self.data.dtypes[self.data.dtypes == 'R'].keys() )
        self.metadata_keys += list(self.data.dtypes[self.data.dtypes == 'B'].keys() )
        if self.data.file_id.max() > 0:
            self.metadata_keys += ['file_id']
        self.metadata_keys += ['dHKL']
        self.compute_dHKL()

        if dmin is not None:
            self.data = self.data[self.data.dHKL >= dmin]

        if isigi_cutoff is not None:
            isigi = self.data[self.intensity_key] / self.data[self.sigma_intensity_key]
            self.data = self.data[isigi >= isigi_cutoff]


    @classmethod
    def from_mtzs(cls, *filenames, anomalous=False, **kwargs):
        def loader():
            for inFN in filenames:
                yield rs.read_mtz(inFN)
        return cls(loader(), anomalous, **kwargs)

    @classmethod
    def half_datasets_from_mtzs(cls, *filenames, seed=1234, anomalous=False, **kwargs):
        def half_loader(first=True):
            for inFN in filenames:
                ds = rs.read_mtz(inFN)
                np.random.seed(seed)
                bkey = get_first_key_of_type(ds, 'B')
                batch = ds[bkey].unique().to_numpy(dtype=int)
                np.random.shuffle(batch)
                half1,half2 = np.array_split(batch, 2)
                if first:
                    yield ds.loc[ds[bkey].isin(half1)]
                else:
                    yield ds.loc[ds[bkey].isin(half2)]
        return cls(half_loader(True), anomalous, **kwargs), cls(half_loader(False), anomalous, **kwargs)

    def label_multiplicity(self):
        self.data['EPSILON'] = 1.
        for i,sg in enumerate(self.spacegroups):
            idx = self.data['file_id'] == i
            self.data.loc[idx, 'EPSILON'] = rs.utils.compute_structurefactor_multiplicity(self.data[idx].get_hkls(), sg)
        return self

    def label_centrics(self):
        H = self.data.get_hkls()
        self.data['CENTRIC'] = False
        for i,sg in enumerate(self.spacegroups):
            idx = self.data['file_id'] == i
            self.data.loc[idx, 'CENTRIC'] = rs.utils.is_centric(self.data[idx].get_hkls(), sg)
        return self

    def compute_dHKL(self):
        self.data['dHKL'] = 0.
        for i,cell in enumerate(self.cells):
            idx = self.data['file_id'] == i
            self.data.loc[idx, 'dHKL'] = rs.utils.compute_dHKL(self.data[idx].get_hkls(), cell)
        return self

    def remove_sys_absences(self):
        H = self.data.get_hkls()
        self.data.loc[:,'ABSENT'] = False
        for i,sg in enumerate(self.spacegroups):
            idx = self.data['file_id'] == i
            self.data.loc[idx, 'ABSENT'] = rs.utils.is_absent(self.data[idx].get_hkls(), sg)
        self.data = self.data[~self.data.ABSENT]
        assert not self.data.ABSENT.max()
        return self

    def append_reference_data(self, data):
        """Append reference data from an rs.DataSet or an Mtz filename."""
        if isinstance(data, str):
            ds = rs.read_mtz(data)
        elif isinstance(data, rs.DataSet):
            ds = data
            if data.index.names != ['H', 'K', 'L']:
                ds.reset_index().set_index(['H', 'K', 'L'])
        else:
            raise TypeError(f"append_reference_data expected string or rs.DataSet, but received {type(ds)}")
        if self.anomalous:
            ds = ds.stack_anomalous().expand_to_p1()
        else:
            ds = ds.expand_anomalous().expand_to_p1()

        self.data = self.data.join(ds.loc[:,ds.dtypes=='F'].iloc[:,0].rename("REF"), on=['Hobs', 'Kobs', 'Lobs'])
        self.data = self.data.join(ds.loc[:,ds.dtypes=='Q'].iloc[:,0].rename("SIGREF"), on=['Hobs', 'Kobs', 'Lobs'])
        return self
 
    def get_results(self):
        """ returns an iterator over results for each experiment id """
        df = self.data.reset_index()
        results = rs.DataSet()
        results['F'] = self.merger.surrogate_posterior.mean().numpy()
        results['SigF'] = self.merger.surrogate_posterior.stddev().numpy()
        results['N'] = df.groupby('miller_id').size()
        results['H'] = df.groupby('miller_id')['H'].first()  
        results['K'] = df.groupby('miller_id')['K'].first()  
        results['L'] = df.groupby('miller_id')['L'].first()  
        results['experiment_id'] = df.groupby('miller_id')['experiment_id'].first()  
        results.infer_mtz_dtypes(inplace=True)
        results.set_index(['H', 'K', 'L'], inplace=True)
        for i in range(results.experiment_id.max() + 1):
            result = results[results.experiment_id == i]
            del result['experiment_id']
            result.merged=True
            result.spacegroup = self.spacegroups[i]
            result.cell = self.cells[i]
            if self.anomalous:
                result = result.unstack_anomalous()[['F(+)', 'SigF(+)', 'F(-)', 'SigF(-)', 'N(+)', 'N(-)']]
                result.fillna(0., inplace=True)
            yield result

    def _build_merger(self):
        self.merger = VariationalMergingModel(
            self.data['miller_id'].to_numpy().astype(np.int32),
            self.scaling_model,
            self.prior,
            self.likelihood,
            self.surrogate_posterior,
            self.weight_kl,
        )

    def train_model(self, iterations, mc_samples=1, learning_rate=0.001, beta_1=0.8, beta_2=0.95, clip_value=None, use_nadam=False):
        if self.merger is None:
            self._build_merger()

        if use_nadam:
            optimizer = tf.keras.optimizers.Nadam(learning_rate, beta_1=beta_1, beta_2=beta_2)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=beta_1, beta_2=beta_2)

        losses = self.merger.fit(optimizer, iterations, s=mc_samples, clip_value=clip_value)
        return losses

    def _add_reference_prior(self, priorfun, reference_f_key="REF", reference_sigf_key="SIGREF"):
        f = self.data.groupby('miller_id').first()[reference_f_key].to_numpy().astype(np.float32)
        sigf = self.data.groupby('miller_id').first()[reference_sigf_key].to_numpy().astype(np.float32)
        self.prior = priorfun(f, sigf)

    def add_folded_normal_posterior(self):
        """
        Use a folded normal for the surrogate posterior (variational distribution).
        This must be called after a prior has been added. 
        """
        if self.prior is None:
            raise(ValueError("self.prior is None, but a prior is needed to intialize the surrogate."))
        from careless.models.merging.surrogate_posteriors import FoldedNormal
        import tensorflow_probability as tfp
        centric = self.data.groupby('miller_id').first().CENTRIC.to_numpy().astype(np.bool)
        low = tf.zeros(len(centric), dtype=tf.float32) + (1. - centric) * tf.math.nextafter(0., 1.)
        high = 1e30
        self.surrogate_posterior = FoldedNormal(
            tfp.util.TransformedVariable(self.prior.mean(), tfp.bijectors.Softplus()),
            tfp.util.TransformedVariable(self.prior.stddev()/10., tfp.bijectors.Softplus()),
            low,
        )

    def add_truncated_normal_posterior(self):
        """
        Use a truncated normal for the surrogate posterior (variational distribution).
        This must be called after a prior has been added. 
        """
        if self.prior is None:
            raise(ValueError("self.prior is None, but a prior is needed to intialize the surrogate."))
        from careless.models.merging.surrogate_posteriors import TruncatedNormal
        import tensorflow_probability as tfp

        centric = self.data.groupby('miller_id').first().CENTRIC.to_numpy().astype(np.bool)
        low = tf.zeros(len(centric), dtype=tf.float32) + (1. - centric) * tf.math.nextafter(0., 1.)
        high = 1e30
        self.surrogate_posterior = TruncatedNormal(
            tfp.util.TransformedVariable(self.prior.mean(), tfp.bijectors.Softplus()),
            tfp.util.TransformedVariable(self.prior.stddev()/10., tfp.bijectors.Softplus()),
            low,
            high,
        )

    def add_rice_woolfson_posterior(self):
        """
        Use a mixed rice and woolfson (folded normal) distribution for the surrogate posterior. 
        This must be called after a prior has been added. 
        """
        if self.prior is None:
            raise(ValueError("self.prior is None, but a prior is needed to intialize the surrogate."))
        from careless.models.merging.surrogate_posteriors import RiceWoolfson
        import tensorflow_probability as tfp
        centric = self.data.groupby('miller_id').first().CENTRIC.to_numpy().astype(np.bool)
        self.surrogate_posterior = RiceWoolfson(
            tfp.util.TransformedVariable(self.prior.mean(), tfp.bijectors.Softplus()),
            tfp.util.TransformedVariable(self.prior.stddev()/10., tfp.bijectors.Softplus()),
            centric
        )

    def add_rice_woolfson_prior(self, reference_f_key='REF', reference_sigf_key='SIGREF'):
        from careless.models.priors.empirical import RiceWoolfsonReferencePrior
        f = self.data.groupby('miller_id').first()[reference_f_key].to_numpy().astype(np.float32)
        sigf = self.data.groupby('miller_id').first()[reference_sigf_key].to_numpy().astype(np.float32)
        centric = self.data.groupby('miller_id').first()['CENTRIC'].to_numpy()
        self.prior = RiceWoolfsonReferencePrior(f, sigf, centric)

    def add_laplace_prior(self, reference_f_key='REF', reference_sigf_key='SIGREF'):
        from careless.models.priors.empirical import LaplaceReferencePrior
        self._add_reference_prior(LaplaceReferencePrior)

    def add_normal_prior(self, reference_f_key='REF', reference_sigf_key='SIGREF'):
        from careless.models.priors.empirical import NormalReferencePrior
        self._add_reference_prior(NormalReferencePrior)

    def add_studentt_prior(self, dof, reference_f_key='REF', reference_sigf_key='SIGREF'):
        from careless.models.priors.empirical import StudentTReferencePrior
        self._add_reference_prior(lambda x,y : StudentTReferencePrior(x, y, dof))

    def add_wilson_prior(self, b=None):
        from careless.models.priors.wilson import WilsonPrior
        centric = self.data.groupby('miller_id').first().CENTRIC.to_numpy().astype(np.float32)
        epsilon = self.data.groupby('miller_id').first().EPSILON.to_numpy().astype(np.float32)
        if b is not None:
            """ 
            Wherein we compute the resolution dependent scale factors with an arbitrary user
            supplied b-factor. This is to make our output look like traditionally scaled
            x-ray data. 

            Σ = exp(-0.25 * b * dHKL**-2.)
            """
            dHKL = self.data.groupby('miller_id').first().dHKL.to_numpy().astype(np.float32)
            sigma = np.exp(-0.25 * b * dHKL**-2.)
        else:
            sigma = 1.
        self.prior = WilsonPrior(centric, epsilon, sigma)

    def add_image_scaler(self, image_id_key='image_id', prior=None):
        """
        Paramters
        ---------
        image_id_key : str (optional)
            Key to use as the image identifier. 
        prior : float (optional)
            The fractional width of the normal prior distribution on image scales. 
            Use this if you want a variational image scaling model.
        """
        from careless.models.scaling.image import ImageScaler,VariationalImageScaler
        if self.scaling_model is None:
            self.scaling_model = []

        if prior is None:
            self.scaling_model.append(ImageScaler(self.data[image_id_key].to_numpy().astype(np.int64)))
        else:
            from tensorflow_probability import distributions as tfd
            prior = tfd.Normal(1., prior)
            self.scaling_model.append(VariationalImageScaler(self.data[image_id_key].to_numpy().astype(np.int64), prior))

    def add_scaling_model(self, layers=20, metadata_keys=None, inverse_square_dHKL=True):
        """
        Parameters
        ----------
        layers : int
            Sequential dense leaky relu layers. The default is 20.
        metadata_keys : list
            List of keys to use for generating the metadata. If None, self.metadata_keys will be used.
        invert_dHKL : bool (optional)
            Optionally transform any metadata keys named 'dHKL' by raising them to the negative 2 power.
            The default is True. 
        """
        if metadata_keys is None:
            metadata_keys = self.metadata_keys
        elif isinstance(metadata_keys, list):
            self.metadata_keys = metadata_keys
        else:
            raise TypeError("metadata_keys has type None but list expected.")
        metadata = self.data[metadata_keys].to_numpy().astype(np.float32)
        if inverse_square_dHKL and 'dHKL' in metadata_keys:
            idx = np.where(np.array(metadata_keys) == 'dHKL')
            metadata[:,idx] = metadata[:,idx]**-2.
        metadata = (metadata - metadata.mean(0))/metadata.std(0)
        from careless.models.scaling.nn import SequentialScaler
        if self.scaling_model is None:
            self.scaling_model = [SequentialScaler(metadata, layers)]
        else:
            self.scaling_model.append(SequentialScaler(metadata, layers))

class HarmonicDeconvolutionMixin:
    def expand_harmonics(self, dmin=None, wavelength_key='Wavelength', wavelength_range=None):
        from careless.utils.laue import expand_harmonics
        self.compute_dHKL() #Make sure this is up to date

        if wavelength_range is None:
            lambda_min = self.data[wavelength_key].min()
            lambda_max = self.data[wavelength_key].max()
        else:
            lambda_min, lambda_max = wavelength_range

        self.data = expand_harmonics(self.data, dmin=dmin, wavelength_key='Wavelength')
        self.data = self.data[(self.data[wavelength_key] >= lambda_min) & (self.data[wavelength_key] <= lambda_max)]
        self.remove_sys_absences()
        return self

class PolyMerger(BaseMerger, HarmonicDeconvolutionMixin):
    def prep_indices(self, separate_files=False, image_id_key=None, experiment_id_key='file_id'):
        """
        Parameters
        ----------
        separate_files : bool
            Default is False. If True, miller indices originating from different input files will be kept separate.
        image_id_key : str
            Key used to identify which image an observation originated from. Default is to use the first 'BATCH' key. 
        file_id_key : str
            Key used to identify which image an observation originated from. 
            Default is 'file_id' which is populated by the MergerBase.from_isomorphous_mtzs constructor. 
        """
        if image_id_key is None:
            image_id_key = get_first_key_of_type(self.data, 'B')

        #This is for merging equivalent millers accross mtzs
        self.data['null'] = self.data.isnull().any(axis=1)

        # Any observation that contains a constituent harmonic missing reference data must be removed en bloc
        # This is a quick way of doing that without invoking groupby.filter with a lambda (very slow)
        obs_group_keys = ['H_0', 'K_0', 'L_0', image_id_key, experiment_id_key]
        idx = self.data[obs_group_keys + ['null']].groupby(obs_group_keys).transform('any').to_numpy()
        self.data = self.data[~idx]
        del(self.data['null'])

        if separate_files:
            self.data['miller_id'] = self.data.groupby(['H', 'K', 'L', experiment_id_key]).ngroup() 
            self.data['experiment_id'] = self.data[experiment_id_key]
        else:
            self.data['miller_id'] = self.data.groupby(['H', 'K', 'L']).ngroup() 
            self.data['experiment_id'] = 0

        self.data['image_id'] = self.data.groupby([image_id_key, 'experiment_id']).ngroup()
        self.data['ray_id'] = self.data.groupby(['H_0','K_0', 'L_0']).ngroup()
        self.data['observation_id'] = self.data.groupby(['ray_id', 'image_id']).ngroup()

        self.data = self.data
        self.label_centrics()
        self.label_multiplicity()
        self.compute_dHKL()
        #self.data['dHKL'] = self.data.dHKL**-2.
        return self

    def _add_likelihood(self, likelihood_func, use_weights=False):
        iobs    = self.data.groupby('observation_id').first()[self.intensity_key].to_numpy().astype(np.float32)
        sigiobs = self.data.groupby('observation_id').first()[self.sigma_intensity_key].to_numpy().astype(np.float32)
        harmonic_id = self.data.observation_id.to_numpy().astype(np.int32)

        if use_weights:
            weights = 1. / sigiobs
            weights = weights/np.mean(weights)
        else:
            weights = None

        self.likelihood = likelihood_func(iobs, sigiobs, harmonic_id, weights)

    def add_normal_likelihood(self, use_weights=False):
        from careless.models.likelihoods.laue import NormalLikelihood
        self._add_likelihood(NormalLikelihood, use_weights)

    def add_laplace_likelihood(self, use_weights=False):
        from careless.models.likelihoods.laue import LaplaceLikelihood
        self._add_likelihood(LaplaceLikelihood, use_weights)

    def add_studentt_likelihood(self, dof, use_weights=False):
        from careless.models.likelihoods.laue import StudentTLikelihood
        self._add_likelihood(lambda x,y,z,w : StudentTLikelihood(x, y, z, dof, w), use_weights)

class MonoMerger(BaseMerger):
    def prep_indices(self, separate_files=False, image_id_key=None, experiment_id_key='file_id'):
        """
        Parameters
        ----------
        separate_files : bool
            Default is False. If True, miller indices originating from different input files will be kept separate.
        image_id_key : str
            Key used to identify which image an observation originated from. Default is 'BATCH'. 
        file_id_key : str
            Key used to identify which image an observation originated from. 
            Default is 'file_id' which is populated by the MergerBase.from_isomorphous_mtzs constructor. 
        """
        if image_id_key is None:
            image_id_key = get_first_key_of_type(self.data, 'B')

        #This is for merging equivalent millers accross mtzs
        self.data.dropna(inplace=True) #There will be nans if reference data were added
        self.remove_sys_absences()
        if separate_files:
            self.data['miller_id'] = self.data.groupby(['H', 'K', 'L', experiment_id_key]).ngroup() 
            self.data['experiment_id'] = self.data[experiment_id_key]
        else:
            self.data['miller_id'] = self.data.groupby(['H', 'K', 'L']).ngroup() 
            self.data['experiment_id'] = 0
        self.data['image_id'] = self.data.groupby([image_id_key, experiment_id_key]).ngroup()
        self.data['observation_id'] = self.data.groupby(['miller_id', 'image_id']).ngroup()
        self.label_centrics()
        self.label_multiplicity()
        self.compute_dHKL()
        #self.data['dHKL'] = self.data.dHKL**-2.
        return self

    def _add_likelihood(self, likelihood_func, use_weights=False):
        iobs = self.data[self.intensity_key].to_numpy().astype(np.float32)
        sigiobs = self.data[self.sigma_intensity_key].to_numpy().astype(np.float32)

        if use_weights:
            weights = 1./sigiobs
            weights = weights/np.mean(weights)
        else:
            weights = None

        self.likelihood = likelihood_func(iobs, sigiobs, weights)

    def add_normal_likelihood(self, use_weights=False):
        from careless.models.likelihoods.mono import NormalLikelihood
        self._add_likelihood(NormalLikelihood, use_weights)

    def add_laplace_likelihood(self, use_weights=False):
        from careless.models.likelihoods.mono import LaplaceLikelihood
        self._add_likelihood(LaplaceLikelihood, use_weights)

    def add_studentt_likelihood(self, dof, use_weights=False):
        from careless.models.likelihoods.mono import StudentTLikelihood
        self._add_likelihood(lambda x,y,w : StudentTLikelihood(x, y, dof, w), use_weights)

