from careless.stats import cchalf,ccanom,ccpred,rsplit
from tempfile import TemporaryDirectory
from os.path import exists
import pandas as pd
import pytest



@pytest.mark.parametrize("bins", [1, 10])
@pytest.mark.parametrize("method", ["spearman", "pearson"])
def test_rsplit(xval_mtz, method, bins):
    tf = TemporaryDirectory()
    csv = f"{tf.name}/out.csv"
    command = f"-o {csv} -b {bins} {xval_mtz}"

    parser = rsplit.ArgumentParser().parse_args(command.split())

    assert not exists(csv)
    rsplit.run_analysis(parser, show=False)
    assert exists(csv)

    df = pd.read_csv(csv)
    assert len(df) == 3*bins + 1


@pytest.mark.parametrize("bins", [1, 10])
@pytest.mark.parametrize("method", ["spearman", "pearson"])
def test_cchalf(xval_mtz, method, bins):
    tf = TemporaryDirectory()
    csv = f"{tf.name}/out.csv"
    command = f"-o {csv} -b {bins} -m {method} {xval_mtz}"

    parser = cchalf.ArgumentParser().parse_args(command.split())

    assert not exists(csv)
    cchalf.run_analysis(parser, show=False)
    assert exists(csv)

    df = pd.read_csv(csv)
    assert len(df) == 3*bins + 1


@pytest.mark.parametrize("bins", [1, 5])
@pytest.mark.parametrize("method", ["spearman", "pearson"])
def test_ccanom(xval_mtz, method, bins):
    tf = TemporaryDirectory()
    csv = f"{tf.name}/out.csv"
    command = f"-o {csv} -b {bins} {xval_mtz}"

    parser = ccanom.ArgumentParser().parse_args(command.split())

    assert not exists(csv)
    ccanom.run_analysis(parser, show=False)
    assert exists(csv)

    df = pd.read_csv(csv)
    assert len(df) == 3*bins + 1


@pytest.mark.parametrize("bins", [1, 10])
@pytest.mark.parametrize("overall", [True, False])
@pytest.mark.parametrize("method", ["spearman", "pearson"])
@pytest.mark.parametrize("multi", [False, True])
def test_ccpred(predictions_mtz, method, bins, overall, multi):
    tf = TemporaryDirectory()
    csv = f"{tf.name}/out.csv"
    command = f"-o {csv} -b {bins} "
    if overall:
        command = command + ' --overall '
    command = command + f" {predictions_mtz} "

    if multi:
        command = command + f" {predictions_mtz} "
        command = command + f" {predictions_mtz} "

    parser = ccpred.ArgumentParser().parse_args(command.split())

    assert not exists(csv)
    ccpred.run_analysis(parser)
    assert exists(csv)

    df = pd.read_csv(csv)

    if multi and not overall:
        assert len(df) == 6*bins + 1
    else:
        assert len(df) == 2*bins + 1

