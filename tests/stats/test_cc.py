from careless.stats import cchalf,ccanom,ccpred,rsplit,image_cc,filter_by_image_cc
from tempfile import TemporaryDirectory
from os.path import exists
from os import symlink
import pandas as pd
import pytest
import reciprocalspaceship as rs



@pytest.mark.parametrize("bins", [1, 10])
@pytest.mark.parametrize("method", ["spearman", "pearson"])
def test_rsplit(xval_mtz, method, bins):
    tf = TemporaryDirectory()
    csv = f"{tf.name}/out.csv"
    png = f"{tf.name}/out.png"
    command = f"-o {csv} -i {png} -b {bins} {xval_mtz}"

    parser = rsplit.ArgumentParser().parse_args(command.split())

    assert not exists(csv)
    assert not exists(png)
    rsplit.run_analysis(parser)
    assert exists(csv)
    assert exists(png)

    df = pd.read_csv(csv)
    assert len(df) == 3*bins 


@pytest.mark.parametrize("bins", [1, 10])
@pytest.mark.parametrize("method", ["spearman", "pearson"])
def test_cchalf(xval_mtz, method, bins):
    tf = TemporaryDirectory()
    csv = f"{tf.name}/out.csv"
    png = f"{tf.name}/out.png"
    command = f"-o {csv} -i {png} -b {bins} -m {method} {xval_mtz}"

    parser = cchalf.ArgumentParser().parse_args(command.split())

    assert not exists(csv)
    assert not exists(png)
    cchalf.run_analysis(parser)
    assert exists(csv)
    assert exists(png)

    df = pd.read_csv(csv)
    assert len(df) == 3*bins 


@pytest.mark.parametrize("bins", [1, 5])
@pytest.mark.parametrize("method", ["spearman", "pearson"])
def test_ccanom(xval_mtz, method, bins):
    tf = TemporaryDirectory()
    csv = f"{tf.name}/out.csv"
    png = f"{tf.name}/out.png"
    command = f"-o {csv} -i {png} -b {bins} {xval_mtz}"

    parser = ccanom.ArgumentParser().parse_args(command.split())

    assert not exists(csv)
    assert not exists(png)
    ccanom.run_analysis(parser)
    assert exists(csv)
    assert exists(png)

    df = pd.read_csv(csv)
    assert len(df) == 3*bins 


@pytest.mark.parametrize("bins", [1, 5])
@pytest.mark.parametrize("overall", [True, False])
@pytest.mark.parametrize("method", ["spearman", "pearson"])
@pytest.mark.parametrize("multi", [False, True])
def test_ccpred(predictions_mtz, method, bins, overall, multi):
    tf = TemporaryDirectory()
    csv = f"{tf.name}/out.csv"
    png = f"{tf.name}/out.png"
    command = f"-o {csv} -i {png} -b {bins} "
    if overall:
        command = command + ' --overall '

    if multi:
        mtz_0 = f'{tf.name}/test_predictions_0.mtz'
        mtz_1 = f'{tf.name}/test_predictions_1.mtz'
        symlink(predictions_mtz, mtz_0)
        symlink(predictions_mtz, mtz_1)
        command = command + f" {mtz_0} "
        command = command + f" {mtz_1} "
    else:
        command = command + f" {predictions_mtz} "

    parser = ccpred.ArgumentParser().parse_args(command.split())

    assert not exists(csv)
    assert not exists(png)
    ccpred.run_analysis(parser)
    assert exists(csv)
    assert exists(png)

    df = pd.read_csv(csv)

    if multi and not overall:
        assert len(df) == 4*bins 
    else:
        assert len(df) == 2*bins


@pytest.mark.parametrize("method", ["spearman", "pearson"])
@pytest.mark.parametrize("multi", [False, True])
def test_image_cc(predictions_mtz, method, multi):
    tf = TemporaryDirectory()
    csv = f"{tf.name}/out.csv"
    png = f"{tf.name}/out.png"
    command = f"-o {csv} -i {png} "

    if multi:
        mtz_0 = f'{tf.name}/test_predictions_0.mtz'
        mtz_1 = f'{tf.name}/test_predictions_1.mtz'
        symlink(predictions_mtz, mtz_0)
        symlink(predictions_mtz, mtz_1)
        command = command + f" {mtz_0} "
        command = command + f" {mtz_1} "
    else:
        command = command + f" {predictions_mtz} "

    parser = image_cc.ArgumentParser().parse_args(command.split())

    assert not exists(csv)
    assert not exists(png)
    image_cc.run_analysis(parser)
    assert exists(csv)
    assert exists(png)

    df = pd.read_csv(csv)

@pytest.mark.parametrize("method", ["spearman", "pearson"])
def test_filter_by_image_cc(predictions_mtz, method, off_file, on_file):
    tf = TemporaryDirectory()
    command = f" {predictions_mtz} {off_file} {on_file} -c 0.1 -o {tf.name}/out"
    out_1 = f"{tf.name}/out_0.mtz"
    out_2 = f"{tf.name}/out_0.mtz"

    assert not exists(out_1)
    assert not exists(out_2)

    parser = filter_by_image_cc.ArgumentParser().parse_args(command.split())
    filter_by_image_cc.run_analysis(parser)

    assert exists(out_1)
    assert exists(out_2)

    rs.read_mtz(out_1)
    rs.read_mtz(out_2)

