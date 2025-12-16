import pytest
import reciprocalspaceship as rs
from tempfile import TemporaryDirectory
from careless.careless import run_careless as _run_careless
from os.path import exists
import multiprocessing

niter = 10
# True will use multiprocessing to bypass the tf memory leak issue,
# but it may result in less verbose error messages.
use_mp = False
eager=True


def run_careless(parser):
    """
    Workaround for tensorflow memory leak.
    see: https://github.com/tensorflow/tensorflow/issues/36465
    """
    if eager:
        parser.run_eagerly=True
    if use_mp:
        proc = multiprocessing.Process(target=_run_careless, args=(parser,))
        proc.start()
        proc.join()
    else:
        _run_careless(parser)

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

@pytest.mark.parametrize("mode", ['mono', 'poly'])
@pytest.mark.parametrize("optimize_r", [False, True])
def test_double_wilson(off_file, on_file, mode, optimize_r):
    flags = f"{mode} --disable-gpu --iterations={niter} dHKL,image_id"
    flags += " --double-wilson-parents=None,0 "
    r_in_range = " --double-wilson-r=0.0,0.9 "
    r_outside_range = " --double-wilson-r=0.0,1.0 "
    if optimize_r:
        flags += " --optimize-double-wilson-r "
    base_test_separate(
        flags + r_in_range, 
        [off_file, on_file]
    )

    with pytest.raises(ValueError):
        base_test_separate(
            flags + r_outside_range, 
            [off_file, on_file]
        )

def test_crystfel(stream_file):
    flags = f"mono --disable-gpu --iterations={niter} --spacegroups=1 dHKL,image_id"
    base_test_together(flags, [stream_file])

    #Careless poly should fail with a clear error message
    with pytest.raises(ValueError):
        flags = f"poly --disable-gpu --iterations={niter} --spacegroups=1 dHKL,image_id"
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

@pytest.mark.parametrize('clip_type', ['--clipvalue', '--clipnorm', '--global-clipnorm'])
def test_clipping(off_file, clip_type):
    """ Test gradient clipping settings """
    with TemporaryDirectory() as td:
        out = td + '/out'
        flags = f"mono --disable-gpu --iterations={niter} {clip_type}=1. dHKL,image_id"
        command = flags +  f" {off_file} {out}"
        from careless.parser import parser
        parser = parser.parse_args(command.split())
        run_careless(parser)

        out_file = out + f"_0.mtz"
        assert exists(out_file)


@pytest.mark.parametrize('scale_bijector', ['exp', 'softplus'])
@pytest.mark.parametrize('image_layers', [None, 2])
def test_scale_bijector(off_file, scale_bijector, image_layers):
    """ Test scale bijector settings """
    with TemporaryDirectory() as td:
        out = td + '/out'
        if image_layers is not None:
            flags = f"mono --disable-gpu --image-layers={image_layers} --iterations={niter} --scale-bijector={scale_bijector} dHKL,image_id"
        else:
            flags = f"mono --disable-gpu --iterations={niter} --scale-bijector={scale_bijector} dHKL,image_id"

        command = flags +  f" {off_file} {out}"
        from careless.parser import parser
        parser = parser.parse_args(command.split())
        run_careless(parser)

        out_file = out + f"_0.mtz"
        assert exists(out_file)

@pytest.mark.parametrize("trainable", [True, False])
@pytest.mark.parametrize("lorentz", [True, False])
@pytest.mark.parametrize("grid_points", [None, 500])
def test_spectral_poly(off_file, trainable, lorentz, grid_points):
    """
    Smoke test for TabulatedSpectralScaler features in poly mode.
    """
    with TemporaryDirectory() as td:
        # 1. Create a dummy spectrum file (Wavelength, Scale)
        # Covering a broad range to ensure test data wavelengths fall inside
        spec_path = td + "/spectrum.txt"
        with open(spec_path, "w") as f:
            f.write("0.0 1.0\n")
            f.write("100.0 1.0\n")

        # 2. Base flags for Laue mode
        # We include 'dHKL' in metadata args as is standard for CLI tests,
        # though the scaler pulls dHKL from the distinct input slot.
        flags = f"poly --disable-gpu --iterations={niter} dHKL,image_id"

        # 3. Add Spectral flags
        flags += f" --spectral-file={spec_path}"

        if trainable:
            flags += " --trainable-spectral-scale"

        if lorentz:
            flags += " --lorentz-correction"

        if grid_points is not None:
            flags += f" --spectral-grid-points={grid_points}"

        # 4. Run test
        # off_file is a standard fixture in test_cli.py providing an MTZ path
        base_test_together(flags, [off_file])

def test_spectral_scale_weight_save_and_load(off_file):
    """
    Test saving and loading weights for the trainable spectral scaler.
    """
    with TemporaryDirectory() as td:
        # 1. Create a dummy spectrum file
        spec_path = td + "/spectrum.txt"
        with open(spec_path, "w") as f:
            f.write("0.0 1.0\n")
            f.write("100.0 1.0\n")

        out = td + '/out'

        # 2. Configure flags for the first run
        # Must use 'poly' mode for spectral scaler features
        # Enable trainable scale so there are variable weights to save
        flags = f"poly --disable-gpu --iterations={niter} dHKL,image_id"
        flags += f" --spectral-file={spec_path} --trainable-spectral-scale"

        command = flags +  f" {off_file} {out}"
        from careless.parser import parser
        parser = parser.parse_args(command.split())

        # Run 1: Training from scratch
        run_careless(parser)

        # Verify output exists
        out_file = out + f"_0.mtz"
        assert exists(out_file)

        # 3. Configure flags for the second run (Loading Weights)
        # Point --scale-file to the weights saved in the previous run
        # Note: TF save_weights uses the prefix, so we pass '{out}_scale'
        flags_load = flags + f" --scale-file={out}_scale"

        out_reloaded = td + '/out_reloaded'
        command = flags_load +  f" {off_file} {out_reloaded}"

        from careless.parser import parser
        parser = parser.parse_args(command.split())

        # Run 2: Training initialized with saved weights
        run_careless(parser)

        # Verify output exists
        out_file_reloaded = out_reloaded + f"_0.mtz"
        assert exists(out_file_reloaded)
