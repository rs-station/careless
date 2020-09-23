from careless.utils.distributions import Stacy,Amoroso
from tensorflow_probability import distributions as tfd
import numpy as np


from careless.utils.device import disable_gpu
status = disable_gpu()
assert status

def compare_dist(ref_dist, test_dist, xmin=1e-3, xmax=10., rtol=1e-4, npoints=1000):
    # Evaluate pdfs
    x = np.linspace(xmin, xmax, npoints)
    p_ref = ref_dist.prob(x[:,None,None]).numpy()
    p_test= test_dist.prob(x[:,None,None]).numpy()

    # Compare pdfs
    assert(np.all(np.isclose(p_ref, p_test, rtol=rtol)))

    # Compare dist.mean
    assert(np.all(np.isclose(ref_dist.mean().numpy(), test_dist.mean().numpy(), rtol=1e-4)))

    # Compare dist.variance
    assert(np.all(np.isclose(ref_dist.variance().numpy(), test_dist.variance().numpy(), rtol=1e-4)))

    # Compare dist.stddev
    assert(np.all(np.isclose(ref_dist.stddev().numpy(), test_dist.stddev().numpy(), rtol=1e-4)))


def test_weibull(xmin=1e-3):
    """
    The Weibull distribution as implemented in TFP has the following pdf:
    ```pdf(x; k, lambda) = (x/lambda)**(k-1) * exp(-(x/lambda)**k) / Z```
    where
    ```
    k = concentration
    lambda = scale
    ````

    The Amoroso distribution as described by Gavin Crooks has the following form:
    ```pdf(x; a, theta, alpha, beta) = ((x-a)/theta)**(alpha*beta - 1) * exp(-((x-a)/theta)**beta)```

    it is clear that:
    ```Weibul(x; k, lambda) = Amoroso(x; 0, lambda, 1, k)```
    """
    # Parameter vectors
    ref_dist_class = tfd.Weibull
    k,lam = np.mgrid[1.1:10:0.1,0.1:10:0.1]
    k,lam = k.astype(np.float32),lam.astype(np.float32)
    ref_params = (k, lam)
    amoroso_params = (0., lam, 1., k)


    # Construct distributions
    ref_dist = ref_dist_class(*ref_params)
    amoroso_dist = Amoroso(*amoroso_params)
    stacy_dist = Stacy(*amoroso_params[1:])

    compare_dist(ref_dist, amoroso_dist)
    compare_dist(ref_dist, stacy_dist)

    kl = stacy_dist.kl_divergence(ref_dist)
    assert np.all(np.isclose(kl, 0., atol=1e-5))

def test_halfnormal(xmin=1e-3, xmax=1e1, npoints=1000):
    """
    The halfnormal distribution as implemented in TFP has the following pdf:
    ```pdf(x; scale) = exp(-0.5 * (x/scale)**2)/Z
    Z = sqrt(2) / (scale * sqrt(pi))
    ```

    ```
    HalfNormal(x; scale) = Amoroso(x; 0, sqrt(2)*scale, 0.5, 2)
                         =   Stacy(x; sqrt(2)*scale, 0.5, 2)
    ```
    """
    ref_dist_class = tfd.HalfNormal
    scale = np.linspace(0.1, 100, 1000).astype(np.float32)
    ref_params = (scale,)
    amoroso_params = (0., np.sqrt(2.) * scale, 0.5, 2.)

    # Construct distributions
    ref_dist = ref_dist_class(*ref_params)
    amoroso_dist = Amoroso(*amoroso_params)
    stacy_dist = Stacy(*amoroso_params[1:])

    compare_dist(ref_dist, amoroso_dist)
    compare_dist(ref_dist, stacy_dist)

    kl = stacy_dist.kl_divergence(ref_dist)
    assert np.all(np.isclose(kl, 0., atol=1e-5))

def test_stacy_kl_div():
    a,b,c = np.mgrid[0.1:5.1:0.5,0.1:5.1:0.5,0.1:5.1:0.5,]
    a,b,c = a.astype(np.float32),b.astype(np.float32),c.astype(np.float32)
    dist = Stacy(a, b, c)
    other = Stacy(a[::-1], b[::-1], c[::-1])

    kl = dist.kl_divergence(dist).numpy()
    assert np.all(np.isclose(kl, 0., atol=1e-5))

    kl = dist.kl_divergence(other).numpy()
    assert np.all(kl >= 0.)


