#Normalized structure factor distributions based on tensorflow probability
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb


def Centric(**kw):
    """
    According to Herr Rupp, the pdf for centric normalized structure factor 
    amplitudes, E, is:
    P(E) = (2/pi)**0.5 * exp(-0.5*E**2)

    In common parlance, this is known as halfnormal distribution with scale=1.0

    RETURNS
    -------
    dist : tfp.distributions.HalfNormal
        Centric normalized structure factor distribution
    """
    dist = tfd.HalfNormal(scale=1., **kw)
    return dist


def Acentric(**kw):
    """
    According to Herr Rupp, the pdf for acentric normalized structure factor 
    amplitudes, E, is:
    P(E) = 2*E*exp(-E**2)

    This is exactly a Raleigh distribution with sigma**2 = 1. This is also
    the same as a Chi distribution with k=2 which has been transformed by 
    rescaling the argument. 

    RETURNS
    -------
    dist : tfp.distributions.TransformedDistribution
        Centric normalized structure factor distribution
    """

    dist = tfd.TransformedDistribution(
        distribution=tfd.Chi(2),
        bijector=tfb.Scale(2**-0.5),
        **kw
    )
    return dist
