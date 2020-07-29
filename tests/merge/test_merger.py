import pytest
from careless.merge.merge import *
from os.path import abspath,dirname,exists


base_dir = dirname(abspath(__file__))
mtz_filenames = [
    base_dir + '/../../data/laue/pyp/off_varEll.mtz',
    base_dir + '/../../data/laue/pyp/2ms_varEll.mtz',
]

reference_filename = base_dir + '/../../data/laue/pyp/pyp_off_phenix.mtz'

for f in mtz_filenames:
    assert exists(f)
assert exists(reference_filename)

reference_data = rs.read_mtz(reference_filename)


@pytest.mark.parametrize("merger_class", [MonoMerger, PolyMerger])
@pytest.mark.parametrize("metadata_keys", [None, ['dHKL', 'X', 'Y', 'BATCH']])
@pytest.mark.parametrize("anomalous", [False, True])
@pytest.mark.parametrize("likelihood", ["Normal", "Laplace", "StudentT"])
@pytest.mark.parametrize("prior", ["Wilson", "Normal", "Laplace", "StudentT"])
def test_MonoMerger_reference_data(merger_class, anomalous, prior, likelihood, metadata_keys):
    merger = merger_class.from_isomorphous_mtzs(*mtz_filenames, anomalous=anomalous)

    if prior != "Wilson":
        #The empirical priors need reference data
        merger.append_reference_data(reference_data)
        assert 'REF' in merger.data
        assert 'SIGREF' in merger.data

    merger.prep_indices()

    if prior == 'Normal':
        merger.add_normal_prior()
    elif prior == 'Laplace':
        merger.add_laplace_prior()
    elif prior == 'StudentT':
        merger.add_studentt_prior(4.)
    elif prior == "Wilson":
        merger.add_wilson_prior()
    else:
        raise ValueError(f"prior, {prior}, is not a valid selection")

    if likelihood == 'Normal':
        merger.add_normal_likelihood()
    elif likelihood == 'Laplace':
        merger.add_laplace_likelihood()
    elif likelihood == 'StudentT':
        merger.add_studentt_likelihood(4.)
    else:
        raise ValueError(f"prior, {prior}, is not a valid selection")


    merger.prior.log_prob(merger.prior.mean())
    merger.likelihood.sample()

    merger.add_scaling_model(layers=1, metadata_keys=metadata_keys)
    merger.scaling_model.sample()

    loss = merger.train_model(10)
    assert np.all(np.isfinite(loss))

    loss = merger.train_model(10, mc_samples=2)
    assert np.all(np.isfinite(loss))
