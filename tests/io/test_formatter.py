import pytest
import reciprocalspaceship as rs
from careless.io.formatter import MonoFormatter
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
@pytest.mark.parametrize('encoding_bit_depth', [1, 3])
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
