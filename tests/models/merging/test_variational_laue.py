from careless.models.priors.empirical import LaplaceReferencePrior,NormalReferencePrior,StudentTReferencePrior
from careless.models.priors.wilson import WilsonPrior
from careless.models.merging.variational import VariationalMergingModel
from careless.models.scaling.nn import MLPScaler
from careless.models.scaling.image import HybridImageScaler,ImageScaler
from careless.models.base import BaseModel
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
import tensorflow_probability as tfp
import pytest
import tensorflow as tf
import numpy as np 

from careless.utils.device import disable_gpu
status = disable_gpu()
assert status




from careless.models.likelihoods.laue import NormalLikelihood,LaplaceLikelihood,StudentTLikelihood
@pytest.mark.parametrize('likelihood_model', [NormalLikelihood, LaplaceLikelihood, StudentTLikelihood])
@pytest.mark.parametrize('prior_model', [LaplaceReferencePrior, NormalReferencePrior, StudentTReferencePrior, WilsonPrior])
@pytest.mark.parametrize('scaling_model', [HybridImageScaler, MLPScaler])
@pytest.mark.parametrize('mc_samples', [3, (), 1])
def test_laue(likelihood_model, prior_model, scaling_model, laue_inputs, mc_samples):
    nrefls = np.max(BaseModel.get_refl_id(laue_inputs)) + 1
    n_images = np.max(BaseModel.get_image_id(laue_inputs)) + 1
    
    #For the students
    dof = 4.
    if likelihood_model == StudentTLikelihood:
        likelihood = likelihood_model(dof)
    else:
        likelihood = likelihood_model()

    if prior_model == WilsonPrior:
        prior = prior_model(
            np.random.choice([True, False], nrefls),
            np.ones(nrefls).astype('float32'),
        )
    elif prior_model == StudentTReferencePrior:
        prior = prior_model(
            np.ones(nrefls).astype('float32'),
            np.ones(nrefls).astype('float32'),
            dof
        )
    else:
        prior = prior_model(
            np.ones(nrefls).astype('float32'),
            np.ones(nrefls).astype('float32'),
        )

    mlp_scaler = MLPScaler(2, 3)
    if scaling_model == HybridImageScaler:
        image_scaler = ImageScaler(n_images) 
        scaler = HybridImageScaler(mlp_scaler, image_scaler)
    elif scaling_model == MLPScaler:
        scaler = mlp_scaler

    surrogate_posterior = tfd.TruncatedNormal(
        tf.Variable(prior.mean()),
        tfp.util.TransformedVariable(
            prior.stddev()/10.,
            tfb.Softplus(),
        ),
        low = 1e-5,
        high = 1e10,
    )

    merger = VariationalMergingModel(surrogate_posterior, prior, likelihood, scaler, mc_samples)
    ipred = merger(laue_inputs)

    isfinite = np.all(np.isfinite(ipred.numpy()))
    assert isfinite

    merger = VariationalMergingModel(surrogate_posterior, prior, likelihood, scaler)
    merger.compile('Adam')
    #merger.fit([l[None,...] for l in laue_inputs], steps_per_epoch=1, batch_size=1)
