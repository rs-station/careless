import reciprocalspaceship as rs
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
import tensorflow as tf
from tqdm import trange
import numpy as np

inFN = "../hewl_ssad/unmerged.mtz"
outFN = "careless_zero_out.mtz"


metadata_keys = [
    'dHKL',
    'XDET',
    'YDET',
    'BATCH',
]

intensity_key = 'IPR'
sigma_intensity_key = 'SIGIPR'

mtz = rs.read_mtz(inFN).compute_dHKL().reset_index()
mtz = mtz[~mtz.label_absences()['ABSENT']]
mtz.loc[:,['Hasu','Kasu','Lasu']] = mtz.hkl_to_asu().loc[:,['H', 'K', 'L']]
mtz['miller_id'] = mtz.groupby(['Hasu', 'Kasu', 'Lasu']).ngroup()

miller_id = mtz['miller_id'].to_numpy()
centric = mtz.label_centrics().groupby('miller_id').first()['CENTRIC'].to_numpy()
multiplicity = mtz.compute_multiplicity().groupby('miller_id').first()['EPSILON'].to_numpy(np.float32)
metadata = mtz[metadata_keys].to_numpy(np.float32)
intensities = mtz[intensity_key].to_numpy(np.float32)
uncertainties = mtz[sigma_intensity_key].to_numpy(np.float32)

###############################################################################
# Below here is verbatim from the Careless manuscript
###############################################################################

steps=10000
n_layers = 20
mc_samples = 3
p_centric  = tfd.HalfNormal(np.sqrt(multiplicity))
p_acentric = tfd.Weibull(2., np.sqrt(multiplicity))

#Construct variational distributions
loc_init   = tf.where(centric, p_centric.mean(), p_acentric.mean())
scale_init = tf.where(centric, p_centric.stddev(), p_acentric.stddev())
q = tfd.TruncatedNormal(
    loc = tf.Variable(loc_init), 
    scale = tfp.util.TransformedVariable(scale_init, tfp.bijectors.Softplus()), 
    low = tf.where(centric, 0., 1e-30),
    high = 1e30,
)

#Construct error model
likelihood = tfd.Normal(loc=intensities, scale=uncertainties)

#Construct scale function
n,d = metadata.shape
NN = tf.keras.models.Sequential()
NN.add(tf.keras.Input(d))
for i in range(n_layers):
    NN.add(tf.keras.layers.Dense(d, kernel_initializer='identity'))
NN.add(tf.keras.layers.Dense(2, kernel_initializer='identity'))

#Evaluate the elbo
def minus_elbo():
    z = q.sample(mc_samples)
    F = tf.gather(z, miller_id, axis=1)
    loc, scale = tf.unstack(NN(metadata), axis=1)
    Sigma = tfd.Normal(loc, scale).sample(mc_samples)
    log_likelihood = tf.reduce_sum(likelihood.log_prob(F * F * Sigma)) 
    log_p_z = tf.where(centric, p_centric.log_prob(z), p_acentric.log_prob(z))
    log_q_z = q.log_prob(z)
    kl_div = tf.reduce_sum(log_q_z - log_p_z)
    return -log_likelihood + kl_div

#Train the model
optimizer = tf.keras.optimizers.Adam()
for i in trange(steps):
    optimizer.minimize(minus_elbo, [q.trainable_variables , NN.trainable_variables])

#Export the results
F,SigF = q.mean().numpy(), q.stddev().numpy()

###############################################################################
# Above here is verbatim from the Careless manuscript
###############################################################################

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
