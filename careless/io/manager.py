import numpy as np
import tensorflow as tf
import reciprocalspaceship as rs
from .asu import ReciprocalASU,ReciprocalASUCollection
from careless.models.base import BaseModel
from careless.models.priors.wilson import WilsonPrior,DoubleWilsonPrior

class DataManager():
    """
    This class comprises various data manipulation methods as well as methods to aid in model construction.
    """
    parser = None
    def __init__(self, inputs, asu_collection, parser=None):
        """
        Parameters
        ----------
        inputs : tuple
        asu_collection : ReciprocalASUCollection
        parser : Namespace (optional)
            A Namespace instance created by careless.parser.parser.parse_args()
        """
        self.inputs = inputs
        self.asu_collection = asu_collection
        self.parser = parser

    @classmethod
    def from_pickle(cls, filename):
        import pickle
        with open(filename, 'rb') as f:
            dm = pickle.load(f)
        return dm

    @classmethod
    def from_mtz_files(cls, filenames, formatter):
        return cls.from_datasets((rs.read_mtz(i) for i in filenames), formatter)

    @classmethod
    def from_stream_files(cls, filenames, formatter):
        return cls.from_datasets((rs.read_crystfel(i) for i in filenames), formatter)

    @staticmethod
    def wilson_sigma(b, dHKL):
        sigma = np.exp(-0.25 * b * np.reciprocal(dHKL*dHKL))
        return sigma

    def get_wilson_sigma(self, b=None):
        if b is None:
            return 1.
        sigma = self.wilson_sigma(b, self.asu_collection.dHKL)
        return sigma

    def get_wilson_prior(self, b=None, k=1.):
        """ Construct a wilson prior with an optional temperature factor, b, appropriate for self.asu_collection. """
        if b is None:
            sigma = 1.
        elif isinstance(b, float):
            sigma = self.get_wilson_sigma(b)
        else:
            raise ValueError(f"parameter b has type{type(b)} but float was expected")
        sigma = sigma * k

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

    def get_predictions(self, model, inputs=None, test_value=0):
        """ 
        Extract results from a surrogate_posterior.

        Parameters
        ----------
        model : VariationalMergingModel
            A merging model from careless
        inputs : tuple (optional)
            Inputs for which to make the predictions if None, self.inputs is used.
        test_value : int (optional)
            Optionally change the value of the `test` column used for crossvalidation.
            The default is 0. 

        Returns
        -------
        predictions : tuple
            A tuple of rs.DataSet objects containing the predictions for each 
            ReciprocalASU contained in self.asu_collection
        """
        laue = BaseModel.is_laue(inputs)

        if inputs is None:
            inputs = self.inputs

        refl_id = BaseModel.get_refl_id(inputs)
        asu_id,H = self.asu_collection.to_asu_id_and_miller_index(refl_id)
        asu_id = asu_id.flatten()
        file_id = model.get_file_id(inputs).flatten()
        image_id = model.get_image_id(inputs).flatten()
        if laue:
            harmonic_id = BaseModel.get_harmonic_id(inputs).flatten()
        else:
            harmonic_id = np.arange(len(refl_id))
        h,k,l = H.T

        output = rs.DataSet({
            'H' : rs.DataSeries(h, dtype='H'),
            'K' : rs.DataSeries(k, dtype='H'),
            'L' : rs.DataSeries(l, dtype='H'),
            'harmonic_id' : rs.DataSeries(harmonic_id, dtype='I'),
            'asu_id'      : rs.DataSeries(asu_id, dtype='I'),
            'image_id'    : rs.DataSeries(image_id, dtype='I'),
            'file_id'     : rs.DataSeries(file_id, dtype='I'),
            'test'        : rs.DataSeries(test_value * np.ones_like(h), dtype='I'),
        }, merged=False).groupby('harmonic_id').first().reset_index()
        del(output['harmonic_id'])

        iobs = BaseModel.get_intensities(inputs).flatten()
        sig_iobs = BaseModel.get_uncertainties(inputs).flatten()
        ipred,sigipred = model.prediction_mean_stddev(inputs)
        scale,sigscale = model.scale_mean_stddev(inputs)

        num_refls = len(output)
        data_cols = {
            'Iobs'    : rs.DataSeries(iobs[:num_refls], dtype='J'),
            'SigIobs' : rs.DataSeries(sig_iobs[:num_refls], dtype='Q'),
            'Ipred'   : rs.DataSeries(ipred[:num_refls], dtype='J'),
            'SigIpred': rs.DataSeries(sigipred[:num_refls], dtype='Q'),
            'Scale'   : rs.DataSeries(scale[:num_refls], dtype='J'),
            'SigScale': rs.DataSeries(sigscale[:num_refls], dtype='Q'),
        }
        for k,v in data_cols.items():
            output[k] = v

        for i,rasu in enumerate(self.asu_collection):
            idx = output['asu_id'] == i
            result = output.loc[idx]
            result.cell = rasu.cell
            result.spacegroup = rasu.spacegroup
            yield result.set_index(['H', 'K', 'L'])


    def get_results(self, surrogate_posterior, inputs=None, output_parameters=True, max_intensity_snr=1e-5):
        """ 
        Extract results from a surrogate_posterior.

        Parameters
        ----------
        surrogate_posterior : tfd.Distribution
            A tensorflow_probability distribution or similar object with `mean` and `stddev` methods
        inputs : tuple (optional)
            Optionally use a different object from self.inputs to compute the redundancy of reflections.
        output_parameters : bool (optional)
            If True, output the parameters of the surrogate distribution in addition to the 
            moments. 
        max_intensity_snr : float (optional)
            The maximum value which will be assigned to I / SigI. 

        Returns
        -------
        results : tuple
            A tuple of rs.DataSet objects containing the results corresponding to each 
            ReciprocalASU contained in self.asu_collection
        """
        if inputs is None:
            inputs = self.inputs
        F = surrogate_posterior.mean().numpy()
        SigF = surrogate_posterior.stddev().numpy()
        I = SigF * SigF + F * F
        # <I^2> = <F^4>
        f4 = surrogate_posterior.moment_4(method='scipy')
        # var(I) = <I^2> - <I>^2 
        #        = <F^4> - <I>^2
        ivar = np.square(I * max_intensity_snr)
        ivar = np.maximum(ivar, f4 - I * I)
        SigI = np.sqrt(ivar)

        params = None
        if output_parameters:
            params = {}
            for k in sorted(surrogate_posterior.parameter_properties()):
                v = surrogate_posterior.parameters[k]
                numpify = lambda x : tf.convert_to_tensor(x).numpy()
                params[k] = numpify(v).flatten() * np.ones(len(F), dtype='float32')
        asu_id,H = self.asu_collection.to_asu_id_and_miller_index(np.arange(len(F)))
        h,k,l = H.T
        refl_id = BaseModel.get_refl_id(inputs)
        N = np.bincount(refl_id.flatten(), minlength=len(F)).astype('float32')
        results = ()
        for i,asu in enumerate(self.asu_collection):
            multiplicity = asu.multiplicity.astype('float32')
            idx = asu_id == i
            idx = idx.flatten()
            output = rs.DataSet({
                'H' : h[idx],
                'K' : k[idx],
                'L' : l[idx],
                'F' : F[idx],
                'SigF' : SigF[idx],
                'I' : I[idx],
                'SigI' : SigI[idx],
                'N' : N[idx],
                }, 
                cell=asu.cell, 
                spacegroup=asu.spacegroup,
                merged=True,
            ).infer_mtz_dtypes().set_index(['H', 'K', 'L'])
            if params is not None:
                for key in sorted(params.keys()):
                    val = params[key]
                    output[key] = rs.DataSeries(val[idx], index=output.index, dtype='R')

            # Remove unobserved refls
            output = output[output.N > 0] 

            # Reformat anomalous data
            if asu.anomalous:
                output = output.unstack_anomalous()
                # PHENIX will expect the sf / error keys in a particular order.
                anom_keys = [
                    'F(+)', 'SigF(+)', 'F(-)', 'SigF(-)', 
                    'I(+)', 'SigI(+)', 'I(-)', 'SigI(-)', 
                    'N(+)', 'N(-)'
                ]
                reorder = anom_keys + [key for key in output if key not in anom_keys]
                output = output[reorder]

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
            #return self.get_tf_dataset(train), self.get_tf_dataset(test)
            return train, test

        test_idx = np.random.random(len(self.inputs[0])) <= test_fraction
        train, test = self.split_mono_data_by_mask(test_idx)
        #return self.get_tf_dataset(train), self.get_tf_dataset(test)
        return train, test

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
                    v = np.pad(v, [[0, len(inv) - len(v)], [0, 0]], constant_values=1.)
                elif name == 'harmonic_id':
                    v = inv[:,None]
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

        #return self.get_tf_dataset(train), self.get_tf_dataset(test)
        return train, test
    # --> end xval data splitting methods

    def build_model(self, parser=None, surrogate_posterior=None, prior=None, likelihood=None, scaling_model=None, mc_sample_size=None):
        """
        Build the model specified in parser, a careless.parser.parser.parse_args() result. Optionally override any of the 
        parameters taken by the VariationalMergingModel constructor.
        The `parser` parameter is required if self.parser is not set. 
        """
        from careless.models.merging.surrogate_posteriors import TruncatedNormal
        from careless.models.merging.variational import VariationalMergingModel
        from careless.models.scaling.image import HybridImageScaler,ImageScaler
        from careless.models.scaling.nn import MLPScaler
        if parser is None:
            parser = self.parser
        if parser is None:
            raise ValueError("No parser supplied, but self.parser is unset")

        if parser.type == 'poly':
            if parser.refine_uncertainties:
                from careless.models.likelihoods.laue import NormalEv11Likelihood as NormalLikelihood
                from careless.models.likelihoods.laue import StudentTEv11Likelihood as StudentTLikelihood
            else:
                from careless.models.likelihoods.laue import NormalLikelihood,StudentTLikelihood
        elif parser.type == 'mono':
            if parser.refine_uncertainties:
                from careless.models.likelihoods.mono import NormalEv11Likelihood as NormalLikelihood
                from careless.models.likelihoods.mono import StudentTEv11Likelihood as StudentTLikelihood
            else:
                from careless.models.likelihoods.mono import NormalLikelihood,StudentTLikelihood

        parents = parser.parents
        r_values = parser.dwr
        if prior is None and parents is None:
            prior = self.get_wilson_prior(parser.wilson_prior_b)
        elif prior is None and parser.parents is not None:
            parents = [None if i == 'None' else int(i) for i in parents.split(',')]
            r_values = [float(i) for i in r_values.split(',')]
            sigma = self.get_wilson_sigma(parser.wilson_prior_b)
            prior = DoubleWilsonPrior(self.asu_collection, parents, r_values, sigma=sigma)

        loc,scale = prior.mean(),prior.stddev()
        scale = scale * parser.structure_factor_init_scale
        low = (1e-32 * ~self.asu_collection.centric).astype('float32')
        if surrogate_posterior is None:
            surrogate_posterior = TruncatedNormal.from_loc_and_scale(loc, scale, low, scale_shift=parser.epsilon)

        if likelihood is None:
            dof = parser.studentt_likelihood_dof
            if dof is None:
                likelihood = NormalLikelihood()
            else:
                likelihood = StudentTLikelihood(dof)

        if scaling_model is None:
            mlp_width = parser.mlp_width
            if mlp_width is None:
                mlp_width = BaseModel.get_metadata(self.inputs).shape[-1]

            if parser.image_layers > 0:
                from careless.models.scaling.image import NeuralImageScaler
                n_images = np.max(BaseModel.get_image_id(self.inputs)) + 1
                scaling_model = NeuralImageScaler(
                    parser.image_layers,
                    n_images,
                    parser.mlp_layers,
                    mlp_width,
                    epsilon=parser.epsilon,
                )
            else:
                mlp_scaler = MLPScaler(parser.mlp_layers, mlp_width, epsilon=parser.epsilon)
                if parser.use_image_scales:
                    n_images = np.max(BaseModel.get_image_id(self.inputs)) + 1
                    image_scaler = ImageScaler(n_images)
                    scaling_model = HybridImageScaler(mlp_scaler, image_scaler)
                else:
                    scaling_model = mlp_scaler

        from tensorflow_probability import distributions as tfd
        model = VariationalMergingModel(surrogate_posterior, prior, likelihood, scaling_model, parser.mc_samples)

        opt = tf.keras.optimizers.Adam(
            parser.learning_rate,
            parser.beta_1,
            parser.beta_2,
        )

        model.compile(
            opt,
            run_eagerly=parser.run_eagerly,
        )
        return model
