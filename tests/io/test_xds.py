import reciprocalspaceship as rs
from careless.io import xds
from tempfile import TemporaryDirectory
import pytest
from os.path import abspath, dirname, join, exists
from shlex import split


@pytest.mark.parametrize("cell", [None, "10 20 30 90 90 90"])
@pytest.mark.parametrize("spacegroup", [None, "19", '"P 21 21 21"'])
@pytest.mark.parametrize("file_type", [None, "integrate"])
def test_integrate_hkl(cell, spacegroup, file_type):
    datadir = join(abspath(dirname(__file__)), "../data")
    hkl_file = join(datadir, "INTEGRATE.HKL")

    tf = TemporaryDirectory()
    mtz_out = f"{tf.name}/out.mtz"

    command = ""
    if cell is not None:
        command += f" --cell {cell} "
    if spacegroup is not None:
        command += f' --spacegroup {spacegroup} '
    if file_type is not None:
        command += f' --file-type "{file_type}" '
    command += f" {hkl_file} {mtz_out}"

    parser = xds.ArgumentParser().parse_args(split(command))

    assert not exists(mtz_out)
    if file_type == 'ascii':
        with pytest.raises(Exception):
            xds.run(parser)
    else:
        xds.run(parser)
        assert exists(mtz_out)
        ds = rs.read_mtz(mtz_out)

@pytest.mark.parametrize("cell", [None, "10 20 30 90 90 90"])
@pytest.mark.parametrize("spacegroup", [None, "19", '"P 21 21 21"'])
@pytest.mark.parametrize("file_type", [None, "ascii"])
def test_ascii_hkl(cell, spacegroup, file_type):
    datadir = join(abspath(dirname(__file__)), "../data")
    hkl_file = join(datadir, "XDS_ASCII.HKL")

    tf = TemporaryDirectory()
    mtz_out = f"{tf.name}/out.mtz"

    command = ""
    if cell is not None:
        command += f" --cell {cell} "
    if spacegroup is not None:
        command += f' --spacegroup {spacegroup} '
    if file_type is not None:
        command += f' --file-type "{file_type}" '
    command += f" {hkl_file} {mtz_out}"

    parser = xds.ArgumentParser().parse_args(split(command))

    assert not exists(mtz_out)
    xds.run(parser)
    assert exists(mtz_out)
    ds = rs.read_mtz(mtz_out)

