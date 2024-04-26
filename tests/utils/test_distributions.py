from careless.utils.distributions import FoldedNormal,Rice
from tensorflow_probability import distributions as tfd
from scipy.stats import foldnorm
import numpy as np


from careless.utils.device import disable_gpu
status = disable_gpu()
assert status

def test_folded_normal(log_min=-3, log_max=3, snr_range=20., npoints=10, dtype='float32'):
    loc = scale = np.logspace(log_min, log_max, log_max-log_min+1)
    loc,scale = np.meshgrid(loc, scale)
    loc,scale = loc.flatten(), scale.flatten()
    loc,scale = loc.astype(dtype),scale.astype(dtype)
    idx = loc / scale <= snr_range
    loc,scale = loc[idx],scale[idx]
    xmin, xmax = loc - snr_range * scale, loc + snr_range * scale
    xmin = np.maximum(xmin, 0.)
    x = np.linspace(xmin, xmax, npoints).astype(dtype)

    q = FoldedNormal(loc, scale)
    result = q.log_prob(x).numpy()
    expected = foldnorm.logpdf(x, loc/scale, scale=scale)
    assert np.allclose(result, expected)

    result = q.mean()
    expected = foldnorm.mean(loc/scale, scale=scale)
    assert np.allclose(result, expected)

    result = q.stddev()
    expected = foldnorm.std(loc/scale, scale=scale)
    assert np.allclose(result, expected)

    result = q.moment(4)
    expected = foldnorm.moment(4, loc/scale, scale=scale)
    assert np.allclose(result, expected)

