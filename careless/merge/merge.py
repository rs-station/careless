import gemmi
import reciprocalspaceship as rs


class HarmonicDeconvolutionMixin:
    def expand_harmonics(self, dmin=None, wavelength_key='Wavelength', wavelength_range=None):
        from careless.utils.laue import expand_harmonics
        expanded = expand_harmonics(self, dmin=None, wavelength_key='Wavelength')
        if wavelength_range is None:
            lambda_min = self[wavelength_key].min()
            lambda_max = self[wavelength_key].max()
        else:
            lambda_min, lambda_max = wavelength_range

        self.data = expanded[(expanded[wavelength_key] >= lambda_min) & (expanded[wavelength_key] <= lambda_max)]
        return self

class MergerBase():
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
    spacegroup = None

    def __init__(self, dataset):
        self.data = dataset
        return self

    @staticmethod
    def from_isomorphous_mtzs(*filenames):
        from careless.utils.io import load_isomorphous_mtzs
        return MergerBase(load_isomorphous_mtzs(*filenames))

    def append_reference_data(self, mtz_filename):
        pass

    def set_merging_spacegroup(sg):
        """
        Parameters
        ----------
        sg : gemmi.SpaceGroup or str or int
        """
        if isinstance(sg, str):
            self.spacegroup = gemmi.SpaceGroup(sg)
        elif isinstance(sg, int):
            self.spacegroup = gemmi.SpaceGroup(sg)
        elif isinstance(sg, gemmi.SpaceGroup):
            self.spacegroup = sg
        else:
            raise ValueError(f"Set_merging_spacegroup received unexpected argument type {type(sg)}")

    def get_results(self):
        df = self.data.reset_index()
        results = rs.DataSet(cell = self.data.cell, spacegroup = self.data.spacegroup)
        results['F'] = self.merger.surrogate_posterior.mean()
        results['SigF'] = self.merger.surrogate_posterior.stddev()
        results['dHKL'] = df.groupby('miller_id').first()['dHKL']
        results['H'] = df.groupby('miller_id')['H'].first()  
        results['K'] = df.groupby('miller_id')['K'].first()  
        results['L'] = df.groupby('miller_id')['L'].first()  
        results['experiment_id'] = df.groupby('miller_id')['experiment_id'].first()  
        keys = ['H', 'K', 'L', 'F', 'SigF', 'experiment_id']
        results = rs.DataSet(results[keys], spacegroup=data.spacegroup, cell=data.cell) 
        results.infer_mtz_dtypes(inplace=True)
        results.set_index(['H', 'K', 'L'], inplace=True)
        return results

    def prep_indices(self, image_id_key='BATCH', experiment_id_key='file_id', anomalous=False):
        df = self.data.hkl_to_asu().reset_index()
        if anomalous:
            friedel_sign = 2 * (data['M/ISYM'] %2 - 0.5).to_numpy()
            self.data.loc[:,['H', 'K', 'L']] = friedel_sign[:,None] * self.data.loc[:,['H', 'K', 'L']]
        #This is for merging equivalent millers accross mtzs
        df['miller_id'] = df.groupby(['H', 'K', 'L']).ngroup() 
        df['image_id'] = df.groupby([image_id_key, experiment_id_key]).ngroup()
        df['observation_id'] = df.groupby(['miller_id', 'image_id']).ngroup()
        self.data = df
        return self

    def train_model(self, prior_name, intensity_key='I', sigma_key='SigI', ...):
        self.prep_indices()

        if df.experiment_id.max() > 0:
            metadata = df[metadata_keys + ['experiment_id']].to_numpy().astype(np.float32) 
        else:
            metadata= df[metadata_keys].to_numpy().astype(np.float32) 

        metadata = (metadata - metadata.mean(0))/metadata.std(0)

        iobs = df.groupby('observation_id').first()[intensity_key].to_numpy().astype(np.float32)
        sigiobs = df.groupby('observation_id').first()[sigma_key].to_numpy().astype(np.float32)

        epsilons = df.groupby('miller_id').first()['epsilon'].to_numpy().astype(np.float32)
        centric = df.groupby('miller_id').first()['CENTRIC'].to_numpy().astype(np.float32)
        scaling_model = SequentialScaler(metadata, layers=20) #Gotta make this a parser argument

        harmonic_id = df['observation_id'].to_numpy().astype(np.int64)
        if parser.studentt:
            likelihood = StudentTLikelihood(iobs, sigiobs, harmonic_id, parser.studentt_dof)
        elif parser.laplace:
            likelihood = LaplaceLikelihood(iobs, sigiobs, harmonic_id)
        else:
            likelihood = NormalLikelihood(iobs, sigiobs, harmonic_id)

        if parser.laplace_prior:
            prior = LaplaceReferencePrior(Fobs, SigFobs)
        elif parser.normal_prior:
            prior = NormalReferencePrior(Fobs, SigFobs)
        else:
            prior = WilsonPrior(centric, epsilons)

        self.merger = VariationalMergingModel(
            df['miller_id'].to_numpy().astype(np.int64),
            [scaling_model],
            prior,
            likelihood,
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate)#, clipvalue=.1)
        print(f"{'#'*80}")
        print(f'Optimizing model')
        print(f"{'#'*80}")
        losses = merger.fit(optimizer, maxiter, s=parser.mc_iterations)

        self.results = self.get_results()


class MultiExperimentMerger(MergerBase):
    def prep_indices(self, image_id_key='BATCH', experiment_id_key='file_id'):
        df = self.data.reset_index()
        #This is for merging equivalent millers accross mtzs
        df['miller_id'] = df.groupby(['H', 'K', 'L', experiment_id_key]).ngroup() 
        df['image_id'] = df.groupby([image_id_key, experiment_id_key]).ngroup()
        df['observation_id'] = df.groupby(['miller_id', 'image_id']).ngroup()

class LaueMerger(MergerBase, HarmonicDeconvolutionMixin):
    def prep_indices(self, image_id_key='BATCH', experiment_id_key='file_id'):
        df = self.data.reset_index()
        #This is for merging equivalent millers accross mtzs
        df['miller_id'] = df.groupby(['H', 'K', 'L']).ngroup() 
        df['image_id'] = df.groupby([image_id_key, experiment_id_key]).ngroup()
        df['ray_id'] = df.groupby(['H_0','K_0', 'L_0']).ngroup()
        df['observation_id'] = df.groupby(['ray_id', 'image_id']).ngroup()

class LaueMerger(MergerBase, HarmonicDeconvolutionMixin):
    def prep_indices(self, image_id_key='BATCH', experiment_id_key='file_id'):
        df = self.data.reset_index()
        #This is for merging equivalent millers accross mtzs
        df['miller_id'] = df.groupby(['H', 'K', 'L']).ngroup() 
        df['image_id'] = df.groupby([image_id_key, experiment_id_key]).ngroup()
        df['ray_id'] = df.groupby(['H_0','K_0', 'L_0']).ngroup()
        df['observation_id'] = df.groupby(['ray_id', 'image_id']).ngroup()

