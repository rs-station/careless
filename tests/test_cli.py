import pytest
import reciprocalspaceship as rs
from tempfile import TemporaryDirectory
from careless.careless import run_careless
from os.path import exists

niter = 10

def test_laue_trx(off_file, on_file):
    with TemporaryDirectory() as td:
        out = td + '/out'
        output_suffixes = ["_0.mtz", "_1.mtz"]
        command = f"poly --separate-files --iterations={niter} dHKL,image_id {off_file} {on_file} {out}"
        from careless.parser import parser
        parser = parser.parse_args(command.split())
        run_careless(parser)

        for suffix in output_suffixes:
            assert exists(out + suffix)
            ds = rs.read_mtz(out + suffix)

