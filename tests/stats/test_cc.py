from os import symlink
from os.path import exists
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest
import reciprocalspaceship as rs

from careless.stats import (
    ccanom,
    cchalf,
    ccpred,
    filter_by_image_cc,
    image_cc,
    isigi,
    rsplit,
)


@pytest.mark.parametrize("bins", [1, 5])
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
    assert len(df) == 3 * bins


@pytest.mark.parametrize("bins", [1, 5])
@pytest.mark.parametrize("method", ["spearman", "pearson", "weighted"])
@pytest.mark.parametrize("use_structure_factors", [False, True])
def test_cchalf(xval_mtz, method, bins, use_structure_factors):
    tf = TemporaryDirectory()
    csv = f"{tf.name}/out.csv"
    png = f"{tf.name}/out.png"
    sf = ""
    if use_structure_factors:
        sf = "--use-structure-factors"

    command = f"-o {csv} -i {png} -b {bins} -m {method} {sf} {xval_mtz}"

    parser = cchalf.ArgumentParser().parse_args(command.split())

    assert not exists(csv)
    assert not exists(png)
    cchalf.run_analysis(parser)
    assert exists(csv)
    assert exists(png)

    df = pd.read_csv(csv)
    assert len(df) == 3 * bins


@pytest.mark.parametrize("bins", [1, 5])
@pytest.mark.parametrize("method", ["spearman", "pearson", "weighted"])
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
    assert len(df) == 3 * bins


@pytest.mark.parametrize("bins", [1, 5])
@pytest.mark.parametrize("overall", [True, False])
@pytest.mark.parametrize("method", ["spearman", "pearson", "weighted"])
@pytest.mark.parametrize("multi", [False, True])
def test_ccpred(predictions_mtz, method, bins, overall, multi):
    tf = TemporaryDirectory()
    csv = f"{tf.name}/out.csv"
    png = f"{tf.name}/out.png"
    command = f"-o {csv} -i {png} -b {bins} "
    if overall:
        command = command + " --overall "

    if multi:
        mtz_0 = f"{tf.name}/test_predictions_0.mtz"
        mtz_1 = f"{tf.name}/test_predictions_1.mtz"
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
        assert len(df) == 4 * bins
    else:
        assert len(df) == 2 * bins


@pytest.mark.parametrize("bins", [1, 5])
@pytest.mark.parametrize("overall", [True, False])
@pytest.mark.parametrize("method", ["spearman", "pearson"])
@pytest.mark.parametrize("multi", [False, True])
def test_isigi(predictions_mtz, method, bins, overall, multi):
    tf = TemporaryDirectory()
    csv = f"{tf.name}/out.csv"
    png = f"{tf.name}/out.png"
    command = f"-o {csv} -i {png} -b {bins} "
    if overall:
        command = command + " --overall "

    if multi:
        mtz_0 = f"{tf.name}/out_0.mtz"
        mtz_1 = f"{tf.name}/out_1.mtz"
        symlink(predictions_mtz, mtz_0)
        symlink(predictions_mtz, mtz_1)
        command = command + f" {mtz_0} "
        command = command + f" {mtz_1} "
    else:
        command = command + f" {predictions_mtz} "

    parser = isigi.ArgumentParser().parse_args(command.split())

    assert not exists(csv)
    assert not exists(png)
    isigi.run_analysis(parser)
    assert exists(csv)
    assert exists(png)

    df = pd.read_csv(csv)

    if multi and not overall:
        assert len(df) == 2 * bins
    else:
        assert len(df) == 1 * bins


@pytest.mark.parametrize("bins", [1, 5])
@pytest.mark.parametrize("overall", [True, False])
def test_isigi_anomalous(merged_mtz, bins, overall):
    """
    Regression test for anomalous input.

    An mtz merged with --anomalous carries I(+)/I(-) rather than a plain
    intensity column, which used to break isigi in two ways:

      1. get_first_key_of_dtype(ds, "J") found nothing and returned None,
         so the run died with `KeyError: None`.
      2. Once stacked, "SigF" precedes "SigI" in column order, so naively
         taking the first "Q" column silently computed I/SigF -- a wrong
         number rather than a crash.

    Guards both: the run must succeed, and the reported value must match
    I/SigI computed independently.
    """
    tf = TemporaryDirectory()
    csv = f"{tf.name}/out.csv"
    png = f"{tf.name}/out.png"
    command = f"-o {csv} -i {png} -b {bins} "
    if overall:
        command = command + " --overall "
    command = command + f" {merged_mtz} "

    parser = isigi.ArgumentParser().parse_args(command.split())

    assert not exists(csv)
    assert not exists(png)
    isigi.run_analysis(parser)
    assert exists(csv)
    assert exists(png)

    df = pd.read_csv(csv)
    assert len(df) == bins
    assert len(df) == len(df.dropna())

    # With a single input file and one bin the reported statistic is just the
    # mean over the whole stacked dataset, so it can be checked directly.
    if bins == 1:
        ds = rs.read_mtz(merged_mtz).stack_anomalous()
        expected = np.mean(ds["I"] / ds["SigI"])
        assert np.isclose(df["I/sigI"].iloc[0], expected)

        # ...and must not be the I/SigF value the naive key lookup produced.
        wrong = np.mean(ds["I"] / ds["SigF"])
        assert not np.isclose(expected, wrong), "fixture cannot detect the bug"
        assert not np.isclose(df["I/sigI"].iloc[0], wrong)


@pytest.mark.parametrize("method", ["weighted", "spearman", "pearson"])
@pytest.mark.parametrize("multi", [False, True])
def test_image_cc(predictions_mtz, method, multi):
    tf = TemporaryDirectory()
    csv = f"{tf.name}/out.csv"
    png = f"{tf.name}/out.png"
    command = f"-o {csv} -i {png} "

    if multi:
        mtz_0 = f"{tf.name}/test_predictions_0.mtz"
        mtz_1 = f"{tf.name}/test_predictions_1.mtz"
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


@pytest.mark.parametrize("method", ["weighted", "spearman", "pearson"])
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
