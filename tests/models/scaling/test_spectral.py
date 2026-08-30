import pytest
import numpy as np
import torch
import reciprocalspaceship as rs
import gemmi
from careless.io.formatter import LaueFormatter
from careless.io.asu import ReciprocalASU, ReciprocalASUCollection
from careless.models.scaling.spectral import TabulatedSpectralScaler
from careless.models.base import BaseModel

class TestSpectralScaler:

    @pytest.fixture
    def mock_dataset(self):
        """
        Create a mock rs.DataSet that mimics the state of data
        just before LaueFormatter.finalize() is called.
        """
        # Create a simple dataset
        # 3 reflections at different wavelengths and resolutions
        data = {
            'H': [1, 2, 3],
            'K': [0, 0, 0],
            'L': [0, 0, 0],
            'intensity': [100.0, 200.0, 300.0],
            'uncertainty': [10.0, 20.0, 30.0],
            'Wavelength': [1.0, 1.5, 2.0], # Angstroms
            'dHKL': [10.0, 5.0, 2.0],      # Angstroms
            'image_id': [0, 0, 0],
            'file_id': [0, 0, 0],
            'asu_id': [0, 0, 0],
            # Helper columns usually added by prep_dataset/finalize
            'H_0': [1, 2, 3],
            'K_0': [0, 0, 0],
            'L_0': [0, 0, 0],
        }

        ds = rs.DataSet(data)
        ds.infer_mtz_dtypes(inplace=True)

        # Set cell/spacegroup (P1 cubic)
        ds.cell = gemmi.UnitCell(100, 100, 100, 90, 90, 90)
        ds.spacegroup = gemmi.SpaceGroup(1)
        return ds

    @pytest.fixture
    def mock_rac(self, mock_dataset):
        """Create a dummy ReciprocalASUCollection."""
        rasu = ReciprocalASU(mock_dataset.cell, mock_dataset.spacegroup, dmin=1.0, anomalous=False)
        return ReciprocalASUCollection([rasu])

    @pytest.fixture
    def mock_formatter(self):
        """Instantiate LaueFormatter with minimal args."""
        return LaueFormatter(
            wavelength_key='Wavelength',
            intensity_key='intensity',
            uncertainty_key='uncertainty',
            image_key='image_id',
            metadata_keys=['dHKL'], # Standard metadata
            separate_outputs=False,
            anomalous=False
        )

    def test_inputs_creation(self, mock_dataset, mock_rac, mock_formatter):
        """
        Verify that LaueFormatter correctly packs 'dHKL' and 'wavelength'
        into the inputs tuple at the correct indices defined by BaseModel.
        """
        # Run finalize to get the inputs tuple
        inputs, _ = mock_formatter.finalize(mock_dataset, mock_rac)

        # 1. Check Wavelength (Standard Laue Input)
        wl_idx = BaseModel.get_index_by_name('wavelength')
        wavelengths = inputs[wl_idx]
        assert np.allclose(np.asarray(wavelengths).flatten(), mock_dataset['Wavelength'].values)

        # 2. Check dHKL (New Input for Lorentz)
        # This asserts that you correctly updated LaueFormatter.finalize to pass 'dHKL'
        try:
            d_idx = BaseModel.get_index_by_name('dHKL')
        except ValueError:
            pytest.fail("BaseModel.input_index does not contain 'dHKL'. Please update BaseModel.")

        d_values = inputs[d_idx]

        # Ensure it matches the inverse-squared metadata values
        assert np.allclose(np.asarray(d_values).flatten(), 1/mock_dataset['dHKL'].values**2)

    def test_scaler_integration_with_lorentz(self, mock_dataset, mock_rac, mock_formatter):
        """
        End-to-End test: Data -> Formatter -> Inputs -> Scaler -> Correct Scale
        """
        # Generate Inputs
        inputs, _ = mock_formatter.finalize(mock_dataset, mock_rac)

        # Setup Tabulated Scaler
        # Spectrum: Flat line at 1.0 to isolate Lorentz effect
        x_grid = np.array([0.5, 2.5])
        y_grid = np.array([1.0, 1.0])

        scaler = TabulatedSpectralScaler(
            x_grid, y_grid,
            lorentz_correction=True,
            num_grid_points=100
        )

        # Run Scaler
        inputs_torch = tuple(torch.as_tensor(np.asarray(x), dtype=torch.float32) for x in inputs)

        scale_dist = scaler(inputs_torch)
        predicted_scale = scale_dist.mean.detach().numpy().flatten()

        # Calculate Expected Lorentz Scales manually
        # L = 4 * lambda^2 * d^2
        wl = mock_dataset['Wavelength'].values
        d = mock_dataset['dHKL'].values
        expected_lorentz = 4.0 * (wl**2) * (d**2)

        # Since spectrum is 1.0, Final Scale == Lorentz Factor
        assert np.allclose(predicted_scale, expected_lorentz, rtol=1e-4)

    def test_scaler_interpolation_with_data(self, mock_dataset, mock_rac, mock_formatter):
        """
        Test combined spectral interpolation + Lorentz on the mock dataset.
        """
        inputs, _ = mock_formatter.finalize(mock_dataset, mock_rac)
        inputs_torch = tuple(torch.as_tensor(np.asarray(x), dtype=torch.float32) for x in inputs)

        # Spectrum: y = 2 * lambda
        x_grid = np.array([0.0, 3.0])
        y_grid = np.array([0.0, 6.0])

        scaler = TabulatedSpectralScaler(
            x_grid, y_grid,
            lorentz_correction=True,
            num_grid_points=1000
        )

        predicted = scaler(inputs_torch).mean.detach().numpy().flatten()

        wl = mock_dataset['Wavelength'].values
        d = mock_dataset['dHKL'].values

        # Expected = Spectrum(wl) * Lorentz(wl, d)
        # Spectrum(wl) = 2 * wl
        # Lorentz = 4 * wl^2 * d^2
        expected = (2 * wl) * (4 * wl**2 * d**2)

        assert np.allclose(predicted, expected, rtol=1e-3)

    def test_trainable_scale_is_learnable(self, mock_dataset, mock_rac, mock_formatter):
        """
        The global multiplier stands in for tfp.util.TransformedVariable with an
        Exp bijector: it must be a parameter, stay positive, start at its
        requested value, and pass gradients.
        """
        inputs, _ = mock_formatter.finalize(mock_dataset, mock_rac)
        inputs_torch = tuple(torch.as_tensor(np.asarray(x), dtype=torch.float32) for x in inputs)

        x_grid = np.array([0.5, 2.5])
        y_grid = np.array([1.0, 1.0])

        fixed = TabulatedSpectralScaler(x_grid, y_grid, num_grid_points=100)
        assert len(list(fixed.parameters())) == 0

        scaler = TabulatedSpectralScaler(
            x_grid, y_grid, trainable_scale=True, initial_value=3.0, num_grid_points=100
        )
        params = list(scaler.parameters())
        assert len(params) == 1
        assert np.isclose(scaler.global_w.item(), 3.0)

        scale = scaler(inputs_torch).mean
        # Flat unit spectrum, no Lorentz -> every scale is the multiplier itself
        assert np.allclose(scale.detach().numpy(), 3.0, rtol=1e-5)

        scale.sum().backward()
        assert params[0].grad is not None
        assert torch.isfinite(params[0].grad).all()
        assert params[0].grad.abs() > 0

    def test_wavelengths_outside_the_table_are_clamped(self, mock_dataset, mock_rac, mock_formatter):
        """
        float_idx is clipped to the table, so wavelengths off either end take the
        nearest tabulated value rather than extrapolating or reading out of bounds.
        """
        inputs, _ = mock_formatter.finalize(mock_dataset, mock_rac)
        inputs_torch = list(torch.as_tensor(np.asarray(x), dtype=torch.float32) for x in inputs)
        wl_idx = BaseModel.get_index_by_name('wavelength')
        # Data wavelengths are 1.0, 1.5, 2.0; put two of them outside [1.2, 1.8]
        inputs_torch[wl_idx] = torch.tensor([[0.1], [1.5], [99.0]], dtype=torch.float32)

        x_grid = np.array([1.2, 1.8])
        y_grid = np.array([10.0, 20.0])
        scaler = TabulatedSpectralScaler(x_grid, y_grid, num_grid_points=1000)

        scale = scaler(tuple(inputs_torch)).mean.detach().numpy()
        assert np.isclose(scale[0], 10.0, rtol=1e-4)   # below the table -> first entry
        assert np.isclose(scale[1], 15.0, rtol=1e-3)   # midpoint interpolates
        assert np.isclose(scale[2], 20.0, rtol=1e-4)   # above the table -> last entry

    def test_output_is_flat_and_deterministic(self, mock_dataset, mock_rac, mock_formatter):
        """
        The scaler returns a point mass shaped (n_obs,), not (n_obs, 1): the
        merging model samples it as q(Sigma) and expects one scale per reflection.
        """
        inputs, _ = mock_formatter.finalize(mock_dataset, mock_rac)
        inputs_torch = tuple(torch.as_tensor(np.asarray(x), dtype=torch.float32) for x in inputs)

        scaler = TabulatedSpectralScaler(
            np.array([0.5, 2.5]), np.array([1.0, 1.0]), num_grid_points=100
        )
        dist = scaler(inputs_torch)

        n_obs = len(mock_dataset)
        assert dist.mean.shape == (n_obs,)
        # rsample adds a leading sample dimension, as Normal.rsample would
        assert dist.rsample((7,)).shape == (7, n_obs)
        # every draw is the same point
        assert torch.equal(dist.rsample((3,))[0], dist.mean)
        # scale is exactly zero, which is what lets deterministic_scale_noise work
        assert torch.equal(dist.scale, torch.zeros_like(dist.mean))


def test_build_model_selects_the_spectral_scaler(tmp_path):
    """
    --spectral-file must actually reach the model. Without this, every CLI test
    above would pass just as well against the default neural network scaler,
    which is a test of the CLI plumbing and nothing else.
    """
    from careless.parser import parser as _parser
    from careless.io.formatter import LaueFormatter
    from careless.io.manager import DataManager
    from os.path import join, dirname, abspath

    spec = tmp_path / "spectrum.txt"
    spec.write_text("0.0 1.0\n100.0 1.0\n")
    off_file = abspath(join(dirname(__file__), "..", "..", "data", "pyp_off.mtz"))

    def build(extra):
        command = (
            f"poly --disable-gpu --iterations=2 dHKL,image_id {extra} "
            f"{off_file} {tmp_path}/out"
        )
        args = _parser.parse_args(command.split())
        df = LaueFormatter.from_parser(args)
        inputs, rac = df.format_files(args.reflection_files)
        return DataManager(inputs, rac, parser=args).build_model(parser=args)

    # Without the flag: whatever the ordinary neural-network path builds
    # (HybridImageScaler by default for poly), but definitely not this one
    assert not isinstance(build("").scaling_model, TabulatedSpectralScaler)

    # With it: the tabulated spectral scaler, and the flags reach the object
    model = build(
        f"--spectral-file={spec} --trainable-spectral-scale --lorentz-correction "
        f"--spectral-grid-points=500"
    )
    scaler = model.scaling_model
    assert isinstance(scaler, TabulatedSpectralScaler)
    assert scaler.lorentz_correction is True
    assert scaler.trainable_scale is True
    assert scaler.y_grid.numel() == 500
    # the learnable multiplier is registered on the merging model, so the
    # optimizer will actually see it
    assert any(p is scaler.global_w_unconstrained for p in model.parameters())
