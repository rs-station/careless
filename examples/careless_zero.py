import reciprocalspaceship as rs
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
import tensorflow as tf
from tqdm import trange
import numpy as np

inFN = "data/hewl_sad/pass1.mtz"
outFN = "careless_zero_out.mtz"

iters=20000

metadata_keys = [
    'dHKL',
    'XDET',
    'YDET',
    'BATCH',
]

intensity_key = 'IPR'
sigma_intensity_key = 'SIGIPR'
n_samples=3

n_layers = 20

mtz = rs.read_mtz(inFN).compute_dHKL().reset_index()
mtz.label_absences()
mtz = mtz[~mtz.label_absences()['ABSENT']]
mtz['Hasu'],mtz['Kasu'],mtz['Lasu'] = mtz.hkl_to_asu()[['H', 'K', 'L']].to_numpy().T
mtz['miller_id'] = mtz.groupby(['Hasu', 'Kasu', 'Lasu']).ngroup()

miller_id = mtz['miller_id'].to_numpy()
centric = mtz.label_centrics().groupby('miller_id').first()['CENTRIC'].to_numpy()
epsilon = mtz.compute_multiplicity().groupby('miller_id').first()['EPSILON'].to_numpy(np.float32)
metadata = mtz[metadata_keys].to_numpy(np.float32)
n,d = metadata.shape

p_centric = tfd.HalfNormal(np.sqrt(epsilon))
p_acentric = tfd.Weibull(2., np.sqrt(epsilon))

loc_init   = tf.where(centric, p_centric.mean(), p_acentric.mean())
scale_init = tf.where(centric, p_centric.stddev(), p_acentric.stddev())

zero = 0.
infinity = 1e30
epsilon = 1e-30

q = tfd.TruncatedNormal(
    loc = tf.Variable(loc_init), 
    scale = tfp.util.TransformedVariable(scale_init, tfb.Softplus()), 
    low = zero,
    high = infinity,
)

likelihood = tfd.Normal(
    mtz[intensity_key].to_numpy(np.float32), 
    mtz[sigma_intensity_key].to_numpy(np.float32), 
)

NN = tf.keras.models.Sequential()
NN.add(tf.keras.Input(d))
for i in range(n_layers):
    NN.add(tf.keras.layers.Dense(d, kernel_initializer='identity'))
NN.add(tf.keras.layers.Dense(2, kernel_initializer='identity'))


def elbo():
    z = q.sample(n_samples) 
    z = tf.maximum(z, epsilon*(1. - centric))
    F = tf.gather(z, miller_id, axis=1)
    loc, scale = tf.unstack(NN(metadata), axis=1)
    Sigma = tfd.Normal(loc, scale).sample(n_samples)
    log_likelihood = tf.reduce_sum(likelihood.log_prob(F * F * Sigma))/n_samples
    kl_div = tf.reduce_sum(
        q.log_prob(z) - \
        tf.where(centric, p_centric.log_prob(z), p_acentric.log_prob(z))
    )/n_samples
    return -log_likelihood + kl_div

opt = tf.keras.optimizers.Adam(0.001)

def train_step():
    opt.minimize(elbo, list(q.trainable_variables) + list(NN.trainable_variables))

for i in trange(iters):
    train_step()

F,SigF = q.mean().numpy(), q.stddev().numpy()

output = rs.DataSet({
    'H' : mtz.groupby('miller_id').first()['Hasu'].astype('H'),
    'K' : mtz.groupby('miller_id').first()['Kasu'].astype('H'),
    'L' : mtz.groupby('miller_id').first()['Lasu'].astype('H'),
    'F' : rs.DataSeries(q.mean().numpy(), dtype='F'),
    'SIGF' : rs.DataSeries(q.stddev().numpy(), dtype='Q'),
    }, 
    cell=mtz.cell, 
    spacegroup=mtz.spacegroup
).set_index(['H', 'K', 'L'])
output.write_mtz(outFN)

from IPython import embed
embed()
