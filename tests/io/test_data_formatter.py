import pytest
import reciprocalspaceship as rs
from careless.io.formatter import MonoFormatter,LaueFormatter
from careless.models.base import BaseModel


# If you change this, you need to leave 'dHKL' at the beginning
metadata_keys = ['dHKL', 'Hobs', 'image_id']

@pytest.mark.parametrize('intensity_key', ['I', None])
@pytest.mark.parametrize('sigma_key', ['SigI', None])
@pytest.mark.parametrize('image_id_key', ['BATCH', None])
@pytest.mark.parametrize('separate_outputs', [True, False])
@pytest.mark.parametrize('anomalous', [True, False])
@pytest.mark.parametrize('dmin', [0., 7.])
@pytest.mark.parametrize('isigi_cutoff', [None, 3.])
@pytest.mark.parametrize('positional_encoding_keys', [None, ['X', 'Y']])
@pytest.mark.parametrize('encoding_bit_depth', [3])
def test_mono_formatter(
        intensity_key,
        sigma_key,
        image_id_key,
        separate_outputs,
        anomalous,
        dmin,
        isigi_cutoff,
        positional_encoding_keys,
        encoding_bit_depth,
        mono_data_set,
    ):
    ds = mono_data_set.copy()
    f = MonoFormatter(
            intensity_key,
            sigma_key,
            image_id_key,
            metadata_keys,
            separate_outputs,
            anomalous,
            dmin,
            isigi_cutoff,
            positional_encoding_keys,
            encoding_bit_depth,
    )
    inputs,rac = f([ds])
    length = None
    for v in inputs:
        assert v.ndim == 2
        assert v.dtype in ('float32', 'int64')
        if length is None:
            length = v.shape[0]
        assert v.shape[0] == length

    metadata = BaseModel.get_metadata(inputs)


def test_mono_formatter_standardizes_metadata_by_default(mono_data_set):
    """With standardize=True (default), non-constant metadata columns must be ≈ z-scored."""
    import numpy as np
    ds = mono_data_set.copy()
    f = MonoFormatter(
        None, None, None,
        ['dHKL', 'Hobs', 'image_id'],
        False, False, 0., None, None, 5,
        standardize=True,
    )
    inputs, _ = f([ds])
    metadata = np.asarray(BaseModel.get_metadata(inputs))

    for col_idx in range(metadata.shape[1]):
        col = metadata[:, col_idx]
        if col.std() > 0:
            assert abs(col.mean()) < 0.1, \
                f"Column {col_idx} mean {col.mean():.4f} not near zero after standardization"
            assert abs(col.std() - 1.0) < 0.1, \
                f"Column {col_idx} std {col.std():.4f} not near 1.0 after standardization"


def test_mono_formatter_no_standardize(mono_data_set):
    """With standardize=False, metadata must NOT be z-scored."""
    import numpy as np
    ds = mono_data_set.copy()
    f_raw = MonoFormatter(
        None, None, None,
        ['dHKL', 'Hobs', 'image_id'],
        False, False, 0., None, None, 5,
        standardize=False,
    )
    f_std = MonoFormatter(
        None, None, None,
        ['dHKL', 'Hobs', 'image_id'],
        False, False, 0., None, None, 5,
        standardize=True,
    )
    inputs_raw, _ = f_raw([ds])
    inputs_std, _ = f_std([ds])
    meta_raw = np.asarray(BaseModel.get_metadata(inputs_raw))
    meta_std = np.asarray(BaseModel.get_metadata(inputs_std))
    # They must differ for at least one column
    assert not np.allclose(meta_raw, meta_std), \
        "Standardized and raw metadata are identical — standardization has no effect"


@pytest.mark.parametrize('lam_min', [None, 0.8])
@pytest.mark.parametrize('lam_max', [None, 1.5])
@pytest.mark.parametrize('intensity_key', ['I', None])
@pytest.mark.parametrize('sigma_key', ['SigI', None])
@pytest.mark.parametrize('image_id_key', ['BATCH', None])
@pytest.mark.parametrize('separate_outputs', [True, False])
@pytest.mark.parametrize('anomalous', [True, False])
@pytest.mark.parametrize('dmin', [None, 7.])
@pytest.mark.parametrize('isigi_cutoff', [None, 3.])
@pytest.mark.parametrize('positional_encoding_keys', [None, ['X', 'Y']])
@pytest.mark.parametrize('encoding_bit_depth', [3])
def test_laue_formatter(
        lam_min,
        lam_max,
        intensity_key,
        sigma_key,
        image_id_key,
        separate_outputs,
        anomalous,
        dmin,
        isigi_cutoff,
        positional_encoding_keys,
        encoding_bit_depth,
        laue_data_set,
    ):
    ds = laue_data_set.copy()
    f = LaueFormatter(
            'Wavelength',
            intensity_key,
            sigma_key,
            image_id_key,
            metadata_keys,
            separate_outputs,
            anomalous,
            lam_min,
            lam_max,
            dmin,
            isigi_cutoff,
            positional_encoding_keys,
            encoding_bit_depth,
    )
    inputs,rac = f([ds])
    length = None
    for v in inputs:
        assert v.ndim == 2
        assert v.dtype in ('float32', 'int64')
        if length is None:
            length = v.shape[0]
        assert v.shape[0] == length

    metadata = BaseModel.get_metadata(inputs)

