from tempfile import TemporaryDirectory
from careless.utils import friedelize,unfriedelize
from os.path import exists
import reciprocalspaceship as rs


def run_friedelize(mtz, directory):
    pfile=directory + "/plus.mtz"
    mfile=directory + "/minus.mtz"
    parser = friedelize.get_parser()
    args = f" {mtz} -p {pfile} -m {mfile} "
    parser = parser.parse_args(args.split())
    friedelize.friedelize(parser)
    return pfile,mfile

def run_careless(pfile, mfile, directory):
    from careless.parser import parser
    from careless.careless import run_careless as _run_careless
    args = f" mono --separate --iterations=10 --disable-gpu --double-wilson-r=0.,0.99 --double-wilson-parents=None,0 dHKL,image_id {pfile} {mfile} {directory}/out"
    parser = parser.parse_args(args.split())
    _run_careless(parser)
    return f"{directory}/out_0.mtz", f"{directory}/out_1.mtz"


def test_friedelize(off_file):
    mtz=off_file
    with TemporaryDirectory() as td:
        pfile,mfile = run_friedelize(mtz, td)
        assert exists(pfile)
        assert exists(mfile)

        ds = rs.read_mtz(mtz)
        plus = rs.read_mtz(pfile)
        minus = rs.read_mtz(mfile)

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

def test_friedelize_and_run_careless(off_file):
    mtz=off_file
    with TemporaryDirectory() as td:
        pfile,mfile = run_friedelize(mtz, td)
        p_merged,m_merged = run_careless(pfile, mfile, td)

        #Checking that a merged mtz exists
        assert exists(p_merged)
        assert exists(m_merged)

        ds_p_merged,ds_m_merged = rs.read_mtz(p_merged),rs.read_mtz(m_merged)
        ds_p,ds_m = rs.read_mtz(pfile),rs.read_mtz(mfile)

        #The counts should be the same as the size of the unmerged data
        assert len(ds_p) + len(ds_m) == ds_p_merged.N.sum() + ds_m_merged.N.sum()

def test_friedelize_run_careless_and_unfriedelize(off_file):
    mtz=off_file
    with TemporaryDirectory() as td:
        pfile,mfile = run_friedelize(mtz, td)
        p_merged,m_merged = run_careless(pfile, mfile, td)

        merged = td + "/merged.mtz"
        args = f" {p_merged} {m_merged} {merged}"
        parser = unfriedelize.get_parser()
        parser = parser.parse_args(args.split())
        unfriedelize.unfriedelize(parser)

        assert exists(merged)
        ds = rs.read_mtz(merged)
        expected_keys = ['F(+)', 'SigF(+)', 'F(-)', 'SigF(-)', 'N(+)', 'N(-)']
        for k in expected_keys:
            assert k in ds

        ds_p_merged = rs.read_mtz(p_merged)
        ds_m_merged = rs.read_mtz(m_merged)
        expected_len = len(ds_p_merged) + len(ds_m_merged)

        assert len(ds.stack_anomalous().dropna()) ==  expected_len

