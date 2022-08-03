import pytest
import reciprocalspaceship as rs
from tempfile import TemporaryDirectory
from careless.careless import run_careless
from os.path import exists

niter = 10

def base_test_separate(flags, filenames):
    with TemporaryDirectory() as td:
        out = td + '/out'
        command = flags +  f" --separate-files {' '.join(filenames)} {out}"
        from careless.parser import parser
        parser = parser.parse_args(command.split())
        run_careless(parser)
        for i,in_file in enumerate(filenames):
            out_file = out + f"_{i}.mtz"
            assert exists(out_file)
            in_ds = rs.read_mtz(in_file)
            out_ds = rs.read_mtz(out_file)
            assert in_ds.spacegroup == out_ds.spacegroup
            if parser.dmin is not None:
                assert out_ds.compute_dHKL().dHKL.min() >= parser.dmin
            if parser.anomalous:
                assert 'F(+)' in out_ds

def base_test_together(flags, filenames):
    with TemporaryDirectory() as td:
        out = td + '/out'
        command = flags +  f" {' '.join(filenames)} {out}"
        from careless.parser import parser
        parser = parser.parse_args(command.split())
        run_careless(parser)

        out_file = out + f"_0.mtz"
        assert exists(out_file)
        out_ds = rs.read_mtz(out_file)
        if parser.dmin is not None:
            assert out_ds.compute_dHKL().dHKL.min() >= parser.dmin
        if parser.anomalous:
            assert 'F(+)' in out_ds

@pytest.mark.parametrize("ev11", [True, False])
@pytest.mark.parametrize("dmin", [None, 7.])
@pytest.mark.parametrize("anomalous", [True, False])
@pytest.mark.parametrize("isigi_cutoff", [None, 1.])
@pytest.mark.parametrize("studentt_dof", [None, 12.])
@pytest.mark.parametrize("separate", [True, False])
@pytest.mark.parametrize("mode", ['mono', 'poly'])
@pytest.mark.parametrize("change_sg", [True, False])
def test_twofile(off_file, on_file, on_file_alt_sg, ev11, dmin, anomalous, isigi_cutoff, studentt_dof, separate, mode, change_sg):
    if change_sg:
        on_file = on_file_alt_sg
    flags = f"{mode} --disable-gpu --iterations={niter} dHKL,image_id"
    if ev11:
        flags += f" --refine-uncertainties "
    if dmin is not None:
        flags += f" --dmin={dmin} "
    if anomalous:
        flags += f" --anomalous "
    if isigi_cutoff is not None:
        flags += f" --isigi-cutoff={isigi_cutoff} "
    if studentt_dof is not None:
        flags += f" --studentt-likelihood-dof={studentt_dof} "
    if separate:
        base_test_separate(flags, [off_file, on_file])
    elif change_sg:
        pass #This is an invalid combination. cannot merge different spacegroups into one file.
    else:
        base_test_together(flags, [off_file, on_file])

def test_crystfel(stream_file):
    flags = f"mono --iterations={niter} --spacegroups=1 dHKL,image_id"
    base_test_together(flags, [stream_file])

def test_scale_weight_save_and_load(off_file):
    """
    Test saving and loading weights. 
    """
    with TemporaryDirectory() as td:
        out = td + '/out'
        flags = f"mono --disable-gpu --iterations={niter} dHKL,image_id"
        command = flags +  f" {off_file} {out}"
        from careless.parser import parser
        parser = parser.parse_args(command.split())
        run_careless(parser)

        out_file = out + f"_0.mtz"
        assert exists(out_file)

        flags = flags + f" --scale-file={out}_scale"
        out = td + '/out_reloaded'
        command = flags +  f" {off_file} {out}"
        from careless.parser import parser
        parser = parser.parse_args(command.split())
        run_careless(parser)
        out_file = out + f"_0.mtz"
        assert exists(out_file)


def test_structure_factor_save_and_load(off_file):
    """
    Test saving and loading weights. 
    """
    with TemporaryDirectory() as td:
        out = td + '/out'
        flags = f"mono --disable-gpu --iterations={niter} dHKL,image_id"
        command = flags +  f" {off_file} {out}"
        from careless.parser import parser
        parser = parser.parse_args(command.split())
        run_careless(parser)

        out_file = out + f"_0.mtz"
        assert exists(out_file)

        flags = flags + f" --structure-factor-file={out}_structure_factor"
        out = td + '/out_reloaded'
        command = flags +  f" {off_file} {out}"
        from careless.parser import parser
        parser = parser.parse_args(command.split())
        run_careless(parser)
        out_file = out + f"_0.mtz"
        assert exists(out_file)

def test_freeze_structure_factor(off_file):
    """ Test `--freeze-structure-factors` for execution """
    with TemporaryDirectory() as td:
        out = td + '/out'
        flags = f"mono --disable-gpu --iterations={niter} --freeze-structure-factors dHKL,image_id"
        command = flags +  f" {off_file} {out}"
        from careless.parser import parser
        parser = parser.parse_args(command.split())
        run_careless(parser)

        out_file = out + f"_0.mtz"
        assert exists(out_file)

def test_freeze_scales(off_file):
    """ Test `--freeze-scales` for execution """
    with TemporaryDirectory() as td:
        out = td + '/out'
        flags = f"mono --disable-gpu --iterations={niter} --freeze-scales dHKL,image_id"
        command = flags +  f" {off_file} {out}"
        from careless.parser import parser
        parser = parser.parse_args(command.split())
        run_careless(parser)

        out_file = out + f"_0.mtz"
        assert exists(out_file)

