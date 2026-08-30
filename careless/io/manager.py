import numpy as np
import torch
import reciprocalspaceship as rs
import gemmi
from .asu import ReciprocalASU, ReciprocalASUCollection
from careless.models.base import BaseModel
from careless.models.priors.wilson import WilsonPrior, DoubleWilsonPrior


class DataManager:
    """
    Organises tensor inputs, constructs model components, and handles train/test splitting.
    """
    parser = None

    def __init__(self, inputs, asu_collection, parser=None):
        """
        Parameters
        ----------
        inputs : tuple of np.ndarray
        asu_collection : ReciprocalASUCollection
        parser : Namespace, optional
            Result of careless.parser.parser.parse_args().
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
        return np.exp(-0.25 * b * np.reciprocal(dHKL * dHKL))

    def get_wilson_sigma(self, b=None):
        if b is None:
            return 1.
        return self.wilson_sigma(b, self.asu_collection.dHKL)

    def get_wilson_prior(self, b=None, k=1.):
        """Construct a WilsonPrior appropriate for self.asu_collection."""
        sigma = self.get_wilson_sigma(b)
        sigma = sigma * k
        return WilsonPrior(
            self.asu_collection.centric,
            self.asu_collection.multiplicity,
            sigma,
        )

    def get_torch_dataset(self, inputs=None):
        """
        Pack inputs into a tuple of torch Tensors (same ordering as the input tuple).

        Parameters
        ----------
        inputs : tuple, optional
            Defaults to self.inputs.

        Returns
        -------
        tuple of torch.Tensor
        """
        if inputs is None:
            inputs = self.inputs
        return tuple(torch.as_tensor(i) for i in inputs)

    def get_predictions(self, model, inputs=None, test_value=0, num_batches=1):
        """
        Extract per-reflection predictions from the model.

        Yields rs.DataSet objects (one per ReciprocalASU).

        Parameters
        ----------
        num_batches : int
            Evaluate the scaling model in this many contiguous chunks rather than
            over the whole dataset at once. This is the inference counterpart of
            train_model's gradient accumulation, and it matters for the same
            reason: ImageLayer's weight gather is O(n_obs * width**2), so the
            prediction pass -- not training -- is what sets the width ceiling once
            accumulation is in use.
        """
        if inputs is None:
            inputs = self.inputs

        laue = BaseModel.is_laue(inputs)

        refl_id = BaseModel.get_refl_id(inputs)
        asu_id, H = self.asu_collection.to_asu_id_and_miller_index(refl_id)
        asu_id = np.asarray(asu_id).flatten()
        file_id = np.asarray(model.get_file_id(inputs)).flatten()
        image_id = np.asarray(model.get_image_id(inputs)).flatten()
        if laue:
            harmonic_id = np.asarray(BaseModel.get_harmonic_id(inputs)).flatten()
        else:
            harmonic_id = np.arange(len(refl_id))
        h, k, l = H.T

        output = rs.DataSet({
            'H'          : rs.DataSeries(h, dtype='H'),
            'K'          : rs.DataSeries(k, dtype='H'),
            'L'          : rs.DataSeries(l, dtype='H'),
            'harmonic_id': rs.DataSeries(harmonic_id, dtype='I'),
            'asu_id'     : rs.DataSeries(asu_id, dtype='I'),
            'image_id'   : rs.DataSeries(image_id, dtype='I'),
            'file_id'    : rs.DataSeries(file_id, dtype='I'),
            'test'       : rs.DataSeries(test_value * np.ones_like(h), dtype='I'),
        }, merged=False)

        _, idx = np.unique(output.harmonic_id, return_index=True)
        output = output.loc[idx].reset_index(drop=True)
        del output['harmonic_id']

        iobs = np.asarray(BaseModel.get_intensities(inputs)).flatten()
        sig_iobs = np.asarray(BaseModel.get_uncertainties(inputs)).flatten()

        torch_inputs = self.get_torch_dataset(inputs)
        device = next(model.parameters()).device
        torch_inputs = tuple(t.to(device) for t in torch_inputs)
        model.eval()
        with torch.no_grad():
            ipred, sigipred = model.prediction_mean_stddev(torch_inputs, num_batches)
            scale, sigscale = model.scale_mean_stddev(torch_inputs, num_batches)

        num_refls = len(output)
        data_cols = {
            'Iobs'    : rs.DataSeries(iobs[:num_refls], dtype='J'),
            'SigIobs' : rs.DataSeries(sig_iobs[:num_refls], dtype='Q'),
            'Ipred'   : rs.DataSeries(ipred[:num_refls], dtype='J'),
            'SigIpred': rs.DataSeries(sigipred[:num_refls], dtype='Q'),
            'Scale'   : rs.DataSeries(scale[:num_refls], dtype='J'),
            'SigScale': rs.DataSeries(sigscale[:num_refls], dtype='Q'),
        }
        for col, val in data_cols.items():
            output[col] = val

        for i, rasu in enumerate(self.asu_collection):
            idx_i = output['asu_id'] == i
            result = output.loc[idx_i]
            result.cell = rasu.cell
            result.spacegroup = rasu.spacegroup
            yield result.set_index(['H', 'K', 'L'])

    def get_results(self, surrogate_posterior, inputs=None, output_parameters=True,
                    max_intensity_snr=1e-5):
        """
        Extract merged structure factor results from the surrogate posterior.

        Yields rs.DataSet objects (one per ReciprocalASU).
        """
        if inputs is None:
            inputs = self.inputs

        F = surrogate_posterior.mean.detach().cpu().numpy()
        SigF = surrogate_posterior.stddev.detach().cpu().numpy()
        I = SigF ** 2 + F ** 2
        f4 = surrogate_posterior.moment_4(method='scipy')
        ivar = np.maximum(np.square(I * max_intensity_snr), f4 - I * I)
        SigI = np.sqrt(ivar)

        params = None
        if output_parameters:
            params = {}
            p_dict = surrogate_posterior.parameters_dict
            for k in sorted(p_dict):
                v = np.asarray(p_dict[k]).flatten()
                params[k] = v * np.ones(len(F), dtype='float32')

        refl_id = np.asarray(BaseModel.get_refl_id(inputs)).flatten()
        asu_id, H = self.asu_collection.to_asu_id_and_miller_index(np.arange(len(F)))
        h, k, l = H.T
        N = np.bincount(refl_id.flatten(), minlength=len(F)).astype('float32')

        results = ()
        for i, asu in enumerate(self.asu_collection):
            multiplicity = asu.multiplicity.astype('float32')
            idx = (asu_id == i).flatten()
            output = rs.DataSet(
                {
                    'H'   : h[idx],
                    'K'   : k[idx],
                    'L'   : l[idx],
                    'F'   : F[idx],
                    'SigF': SigF[idx],
                    'I'   : I[idx],
                    'SigI': SigI[idx],
                    'N'   : N[idx],
                },
                cell=asu.cell,
                spacegroup=asu.spacegroup,
                merged=True,
            ).infer_mtz_dtypes().set_index(['H', 'K', 'L'])

            if params is not None:
                for key in sorted(params):
                    output[key] = rs.DataSeries(params[key][idx], index=output.index, dtype='R')

            output = output[output.N > 0]

            if asu.anomalous:
                output = output.unstack_anomalous()
                anom_keys = [
                    'F(+)', 'SigF(+)', 'F(-)', 'SigF(-)',
                    'I(+)', 'SigI(+)', 'I(-)', 'SigI(-)',
                    'N(+)', 'N(-)'
                ]
                reorder = anom_keys + [key for key in output if key not in anom_keys]
                output = output[reorder]

            results += (output,)
        return results

    # ------------------------------------------------------------------
    # Train / test splitting
    # ------------------------------------------------------------------

    def split_mono_data_by_mask(self, test_idx):
        test, train = (), ()
        for inp in self.inputs:
            test  += (inp[ test_idx.flatten(), ...],)
            train += (inp[~test_idx.flatten(), ...],)
        return train, test

    def split_data_by_refl(self, test_fraction=0.5):
        if BaseModel.is_laue(self.inputs):
            harmonic_id = BaseModel.get_harmonic_id(self.inputs)
            test_idx = (np.random.random(harmonic_id.max() + 1) <= test_fraction)[harmonic_id]
            return self.split_laue_data_by_mask(test_idx)

        test_idx = np.random.random(len(self.inputs[0])) <= test_fraction
        return self.split_mono_data_by_mask(test_idx)

    def split_laue_data_by_mask(self, test_idx):
        harmonic_id = BaseModel.get_harmonic_id(self.inputs)

        isect = np.intersect1d(
            harmonic_id[test_idx].flatten(),
            harmonic_id[~test_idx].flatten(),
        )
        if len(isect) > 0:
            raise ValueError(
                f"test_idx splits harmonic observations with harmonic_id: {isect}"
            )

        def split(inputs, idx):
            harmonic_id = BaseModel.get_harmonic_id(inputs)
            result = ()
            uni, inv = np.unique(harmonic_id[idx], return_inverse=True)
            for i, v in enumerate(inputs):
                name = BaseModel.get_name_by_index(i)
                if name in ('intensities', 'uncertainties'):
                    v = v[uni]
                    v = np.pad(v, [[0, len(inv) - len(v)], [0, 0]], constant_values=1.)
                elif name == 'harmonic_id':
                    v = inv[:, None]
                else:
                    v = v[idx.flatten(), ...]
                result += (v,)
            return result

        return split(self.inputs, ~test_idx), split(self.inputs, test_idx)

    def split_data_by_image(self, test_fraction=0.5):
        image_id = BaseModel.get_image_id(self.inputs)
        test_idx = np.random.random(image_id.max() + 1) <= test_fraction

        if True not in test_idx:
            test_idx[0] = True
        elif False not in test_idx:
            test_idx[0] = False

        test_idx = test_idx[image_id]
        if BaseModel.is_laue(self.inputs):
            return self.split_laue_data_by_mask(test_idx)
        return self.split_mono_data_by_mask(test_idx)

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def build_model(self, parser=None, surrogate_posterior=None, prior=None,
                    likelihood=None, scaling_model=None, mc_sample_size=None):
        """
        Build the VariationalMergingModel specified by parser.
        Any argument may be overridden.
        """
        from careless.distributions import TruncatedNormal
        from careless.models.merging.variational import VariationalMergingModel
        from careless.models.scaling.image import HybridImageScaler, ImageScaler, NeuralImageScaler
        from careless.models.scaling.nn import MLPScaler
        from careless.models.scaling.spectral import TabulatedSpectralScaler

        if parser is None:
            parser = self.parser
        if parser is None:
            raise ValueError("No parser supplied and self.parser is unset.")

        if parser.type == 'poly':
            if parser.refine_uncertainties:
                from careless.models.likelihoods.laue import (
                    NormalEv11Likelihood as NormalLikelihood,
                    StudentTEv11Likelihood as StudentTLikelihood,
                )
            else:
                from careless.models.likelihoods.laue import NormalLikelihood, StudentTLikelihood
        elif parser.type == 'mono':
            if parser.refine_uncertainties:
                from careless.models.likelihoods.mono import (
                    NormalEv11Likelihood as NormalLikelihood,
                    StudentTEv11Likelihood as StudentTLikelihood,
                )
            else:
                from careless.models.likelihoods.mono import NormalLikelihood, StudentTLikelihood

        # Prior
        parents = parser.parents
        r_values = parser.dwr
        if prior is None and parents is None:
            prior = self.get_wilson_prior(parser.wilson_prior_b)
        elif prior is None and parents is not None:
            parents = [None if i == 'None' else int(i) for i in parents.split(',')]
            r_values = [float(i) for i in r_values.split(',')]
            for r in r_values:
                if r >= 1.0 or r <= -1.0:
                    raise ValueError(
                        f"--double-wilson-r value {r} outside of allowed range (-1, 1)"
                    )
                if r < 0:
                    from warnings import warn
                    warn(f"--double-wilson-r value {r} is negative")

            sigma = self.get_wilson_sigma(parser.wilson_prior_b)
            reindexing_ops = parser.reindexing_ops
            if reindexing_ops is not None:
                reindexing_ops = [gemmi.Op(i) for i in reindexing_ops.split(';')]

            prior = DoubleWilsonPrior(
                self.asu_collection, parents, r_values, reindexing_ops,
                sigma=sigma, optimize_r=parser.optimize_double_wilson_r
            )

        # Surrogate posterior
        if surrogate_posterior is None:
            loc = prior.mean.detach().cpu().numpy()
            scale = prior.stddev.detach().cpu().numpy()
            scale = scale * parser.structure_factor_init_scale
            # Asymmetric lower bounds matching TF v0.5.4:
            #   centric     → low = 0.0   (HalfNormal has support [0, ∞))
            #   acentric    → low = 1e-32 (Weibull has strict positive support)
            # WilsonPrior.log_prob already clamps x before evaluating either branch,
            # so centric samples at exactly zero are handled safely.
            centric_mask = np.array(self.asu_collection.centric, dtype=bool)
            low = np.where(centric_mask, 0.0, 1e-32).astype('float32')
            surrogate_posterior = TruncatedNormal.from_loc_and_scale(
                loc, scale, low, scale_shift=parser.epsilon
            )

        # Likelihood
        if likelihood is None:
            dof = parser.studentt_likelihood_dof
            if dof is None:
                likelihood = NormalLikelihood()
            else:
                likelihood = StudentTLikelihood(dof)

        # Scaling model
        if scaling_model is None:
            mlp_width = parser.mlp_width
            if mlp_width is None:
                mlp_width = BaseModel.get_metadata(self.inputs).shape[-1]

            scale_bijector = parser.scale_bijector.lower()

            # Empirical std of observed intensities used to shift the initial
            # predicted scale to the right order of magnitude (mirrors TF v0.5.4
            # tfb.Shift(istd) applied to the scale distribution output).
            intensities = BaseModel.get_intensities(self.inputs).flatten()
            istd = float(np.std(intensities))
            if istd == 0.0:
                istd = 1.0

            if parser.spectral_file is not None:
                # Load the 2-column text file
                data = np.loadtxt(parser.spectral_file)

                # Assuming Col 0 = Wavelength, Col 1 = Scale
                x_grid = data[:, 0]
                y_grid = data[:, 1]

                scaling_model = TabulatedSpectralScaler(
                    x_grid=x_grid,
                    y_grid=y_grid,
                    trainable_scale=parser.trainable_spectral_scale,
                    num_grid_points=parser.spectral_grid_points,
                    lorentz_correction=parser.lorentz_correction,
                )
            elif parser.image_layers > 0:
                n_images = int(np.max(BaseModel.get_image_id(self.inputs))) + 1
                scaling_model = NeuralImageScaler(
                    parser.image_layers,
                    n_images,
                    parser.mlp_layers,
                    mlp_width,
                    epsilon=parser.epsilon,
                    scale_bijector=scale_bijector,
                    scale_multiplier=istd,
                )
            else:
                mlp_scaler = MLPScaler(
                    parser.mlp_layers, mlp_width,
                    epsilon=parser.epsilon,
                    scale_bijector=scale_bijector,
                    scale_multiplier=istd,
                )
                if parser.use_image_scales:
                    n_images = int(np.max(BaseModel.get_image_id(self.inputs))) + 1
                    image_scaler = ImageScaler(n_images)
                    scaling_model = HybridImageScaler(mlp_scaler, image_scaler)
                else:
                    scaling_model = mlp_scaler

        if mc_sample_size is None:
            mc_sample_size = parser.mc_samples

        model = VariationalMergingModel(
            surrogate_posterior,
            prior,
            likelihood,
            scaling_model,
            mc_sample_size=mc_sample_size,
            kl_weight=parser.kl_weight,
            learning_rate=parser.learning_rate,
            beta_1=parser.beta_1,
            beta_2=parser.beta_2,
            clipnorm=parser.clipnorm,
            clipvalue=parser.clipvalue,
            global_clipnorm=parser.global_clipnorm,
            adam_epsilon=parser.adam_epsilon,
            filter_nan_gradients=parser.filter_nan_gradients,
        )

        return model
