import pathlib
from tempfile import TemporaryDirectory
import pytest
from careless.utils import friedel
from os.path import exists
import reciprocalspaceship as rs
from careless.stats import ccanom
import pandas as pd
from careless.parser import parser as careless_parser
from careless.careless import run_careless as _run_careless


def run_split_friedel(mtz, directory):
    pfile = directory / "plus.mtz"
    mfile = directory / "minus.mtz"
    parser = friedel.get_split_friedel_parser()
    flags = f" {mtz} -p {pfile} -m {mfile} "
    args = parser.parse_args(flags.split())
    friedel.split_friedel(args)
    return pfile,mfile

def run_careless(pfile, mfile, directory, extra_flags=''):
    flags = f" mono {extra_flags} --separate --iterations=10 --disable-gpu --double-wilson-r=0.,0.99 --double-wilson-parents=None,0 dHKL,image_id {pfile} {mfile} {directory}/out"
    args = careless_parser.parse_args(flags.split())
    _run_careless(args)
    return directory / "out_0.mtz", directory / "out_1.mtz"

def test_split_friedel_merged(merged_mtz):
    merged_mtz = pathlib.Path(merged_mtz)
    with TemporaryDirectory() as _td, pathlib.Path(_td) as td:
        with pytest.raises(ValueError):
            pfile,mfile = run_split_friedel(merged_mtz, td)

def test_split_friedel(off_file):
    mtz=pathlib.Path(off_file)
    with TemporaryDirectory() as _td, pathlib.Path(_td) as td:
        pfile,mfile = run_split_friedel(mtz, td)
        assert exists(pfile)
        assert exists(mfile)

        ds = rs.read_mtz(str(mtz))
        plus = rs.read_mtz(str(pfile))
        minus = rs.read_mtz(str(mfile))

        #Check that all reflections are still there
        assert len(plus) + len(minus) == len(ds)

        #Check that no reflections are duplicated across files
        assert len(plus.index.intersection(minus.index)) == 0

        #Check that the centrics are in plus
        assert len(minus.centrics) == 0
        assert len(plus.centrics) == len(ds.centrics)

        #Check that all acentrics are in the right half of reciprocal space
        assert (plus.hkl_to_asu().acentrics['M/ISYM'] % 2 == 1).all()
        assert (minus.hkl_to_asu().acentrics['M/ISYM'] % 2 == 0).all()

def test_split_friedel_and_run_careless(off_file):
    mtz=off_file
    with TemporaryDirectory() as _td, pathlib.Path(_td) as td:
        pfile,mfile = run_split_friedel(mtz, td)
        p_merged,m_merged = run_careless(pfile, mfile, td)

        #Checking that a merged mtz exists
        assert exists(p_merged)
        assert exists(m_merged)

        ds_p_merged = rs.read_mtz(str(p_merged))
        ds_m_merged = rs.read_mtz(str(m_merged))

        ds_p = rs.read_mtz(str(pfile))
        ds_m = rs.read_mtz(str(mfile))

        #The counts should be the same as the size of the unmerged data
        assert len(ds_p) + len(ds_m) == ds_p_merged.N.sum() + ds_m_merged.N.sum()

def test_split_merge_and_combine(off_file):
    mtz=off_file
    with TemporaryDirectory() as _td, pathlib.Path(_td) as td:
        pfile,mfile = run_split_friedel(mtz, td)
        extra_flags = "--merge-half-datasets"
        p_merged,m_merged = run_careless(pfile, mfile, td, extra_flags)

        merged = td / "merged.mtz"
        flags = f" {p_merged} {m_merged} {merged}"
        parser = friedel.get_combine_friedel_parser()
        parser = parser.parse_args(flags.split())
        friedel.combine_friedel(parser)

        assert exists(merged)
        ds = rs.read_mtz(str(merged))
        expected_keys = ['F(+)', 'SigF(+)', 'F(-)', 'SigF(-)', 'N(+)', 'N(-)']
        for k in expected_keys:
            assert k in ds

        ds_p_merged = rs.read_mtz(str(p_merged))
        ds_m_merged = rs.read_mtz(str(m_merged))
        expected_len = len(ds_p_merged) + len(ds_m_merged)

        assert len(ds.stack_anomalous().dropna()) ==  expected_len

        #Now check the _xval_{0,1}.mtz used in cchalf calcs
        combined = td / "xval.mtz"
        p_xval = pathlib.Path(str(p_merged).removesuffix('0.mtz') + 'xval_0.mtz')
        m_xval = pathlib.Path(str(m_merged).removesuffix('1.mtz') + 'xval_1.mtz')
        flags = f" {p_xval} {m_xval} {combined}"
        parser = friedel.get_combine_friedel_parser()
        parser = parser.parse_args(flags.split())
        friedel.combine_friedel(parser)

        assert exists(combined)
        ds = rs.read_mtz(str(combined))
        ds_p_xval = rs.read_mtz(str(p_xval))
        ds_m_xval = rs.read_mtz(str(m_xval))
        expected_len = len(ds_p_xval) + len(ds_m_xval)

        #Check the combined xval is the right size
        assert len(ds.stack_anomalous().dropna()) ==  expected_len

        #Is output compatible with ccanom?
        bins = 2
        csv = f"{td}/out.csv"
        png = f"{td}/out.png"
        command = f"-o {csv} -i {png} -b {bins} {combined}"

        parser = ccanom.ArgumentParser().parse_args(command.split())

        assert not exists(csv)
        assert not exists(png)
        ccanom.run_analysis(parser)
        assert exists(csv)
        assert exists(png)

        df = pd.read_csv(csv)
        assert len(df) == bins 
        assert len(df) == len(df.dropna()) #should be no nans

