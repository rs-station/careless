import pytest
import numpy as np
import tensorflow as tf
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
        assert np.allclose(wavelengths.flatten(), mock_dataset['Wavelength'].values)

        # 2. Check dHKL (New Input for Lorentz)
        # This asserts that you correctly updated LaueFormatter.finalize to pass 'dHKL'
        try:
            d_idx = BaseModel.get_index_by_name('dHKL')
        except ValueError:
            pytest.fail("BaseModel.input_index does not contain 'dHKL'. Please update BaseModel.")

        d_values = inputs[d_idx]

        # Ensure it matches the inverse-squared metadata values
        assert np.allclose(d_values.flatten(), 1/mock_dataset['dHKL'].values**2)

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
        # Note: Scaler expects TF tensors, but usually handles numpy arrays via TF auto-casting.
        # If not, convert explicitly:
        inputs_tf = tuple(tf.convert_to_tensor(x, dtype=tf.float32) for x in inputs)

        scale_dist = scaler(inputs_tf)
        predicted_scale = scale_dist.mean().numpy().flatten()

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
        inputs_tf = tuple(tf.convert_to_tensor(x, dtype=tf.float32) for x in inputs)

        # Spectrum: y = 2 * lambda
        x_grid = np.array([0.0, 3.0])
        y_grid = np.array([0.0, 6.0]) 

        scaler = TabulatedSpectralScaler(
            x_grid, y_grid,
            lorentz_correction=True,
            num_grid_points=1000
        )

        predicted = scaler(inputs_tf).mean().numpy().flatten()

        wl = mock_dataset['Wavelength'].values
        d = mock_dataset['dHKL'].values

        # Expected = Spectrum(wl) * Lorentz(wl, d)
        # Spectrum(wl) = 2 * wl
        # Lorentz = 4 * wl^2 * d^2
        expected = (2 * wl) * (4 * wl**2 * d**2)

        assert np.allclose(predicted, expected, rtol=1e-3)
