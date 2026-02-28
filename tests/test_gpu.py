"""
Tests for GPU (CUDA) support.

Each test is marked xfail when no CUDA device is available, so the suite
passes cleanly on CPU-only CI runners. Locally, with a CUDA-enabled PyTorch
build and a GPU, the tests run normally and must pass.

Note: The careless-torch conda environment must have a CUDA-enabled PyTorch
build installed for these tests to exercise GPU paths. Install with:

    pip install torch --index-url https://download.pytorch.org/whl/cu121
"""

import pytest
import torch
import numpy as np

_no_gpu = not torch.cuda.is_available()
_xfail_no_gpu = pytest.mark.xfail(
    _no_gpu,
    reason="No CUDA GPU available (CPU-only PyTorch build or no GPU)",
    strict=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_mono_merger(mono_inputs, device):
    """Construct a minimal mono VariationalMergingModel on *device*."""
    from careless.models.priors.wilson import WilsonPrior
    from careless.models.merging.variational import VariationalMergingModel
    from careless.models.scaling.nn import MLPScaler
    from careless.models.base import BaseModel
    from careless.distributions import TruncatedNormal
    from careless.models.likelihoods.mono import NormalLikelihood

    inputs  = tuple(torch.as_tensor(x) for x in mono_inputs)
    nrefls  = int(BaseModel.get_refl_id(inputs).max()) + 1
    n_images = int(BaseModel.get_image_id(inputs).max()) + 1

    prior   = WilsonPrior(
        np.random.choice([True, False], nrefls),
        np.ones(nrefls, dtype='float32'),
    )
    loc     = prior.mean.detach().numpy()
    scale   = prior.stddev.detach().numpy() / 10.0
    surrogate_posterior = TruncatedNormal.from_loc_and_scale(loc, scale)

    merger  = VariationalMergingModel(
        surrogate_posterior, prior, NormalLikelihood(), MLPScaler(2, 8)
    )
    merger.to(device)
    return merger, inputs


# ---------------------------------------------------------------------------
# Unit-level GPU tests
# ---------------------------------------------------------------------------

@_xfail_no_gpu
def test_parameters_on_cuda(mono_inputs):
    """All model parameters must reside on the CUDA device after model.to()."""
    device  = torch.device('cuda:0')
    merger, _ = _build_mono_merger(mono_inputs, device)
    for name, p in merger.named_parameters():
        assert p.device.type == 'cuda', \
            f"Parameter '{name}' is on {p.device}, expected cuda"


@_xfail_no_gpu
def test_forward_pass_on_cuda(mono_inputs):
    """Forward pass should produce finite predictions on the CUDA device."""
    from careless.models.base import reset_losses_and_metrics, get_accumulated_losses

    device    = torch.device('cuda:0')
    merger, inputs = _build_mono_merger(mono_inputs, device)
    inputs_gpu = tuple(t.to(device) for t in inputs)

    reset_losses_and_metrics()
    ipred = merger(inputs_gpu)
    loss  = sum(get_accumulated_losses())

    assert ipred.device.type == 'cuda', \
        f"ipred is on {ipred.device}, expected cuda"
    assert torch.all(torch.isfinite(ipred)), "ipred contains non-finite values on GPU"
    assert torch.isfinite(loss), "ELBO loss is non-finite on GPU"


@_xfail_no_gpu
def test_backward_pass_on_cuda(mono_inputs):
    """Gradients should be finite for all parameters after backward on CUDA."""
    from careless.models.base import reset_losses_and_metrics, get_accumulated_losses

    device    = torch.device('cuda:0')
    merger, inputs = _build_mono_merger(mono_inputs, device)
    inputs_gpu = tuple(t.to(device) for t in inputs)

    reset_losses_and_metrics()
    merger(inputs_gpu)
    sum(get_accumulated_losses()).backward()

    for name, p in merger.named_parameters():
        if p.grad is not None:
            assert torch.all(torch.isfinite(p.grad)), \
                f"Non-finite gradient for parameter '{name}' on GPU"


@_xfail_no_gpu
def test_train_model_on_cuda(mono_inputs):
    """train_model() should run on CUDA and return finite loss history."""
    device  = torch.device('cuda:0')
    merger, inputs = _build_mono_merger(mono_inputs, device)

    history = merger.train_model(inputs, steps=5, progress=False)

    assert 'NLL' in history, "History missing 'NLL' key"
    assert all(np.isfinite(v) for v in history['NLL']), \
        "NLL history contains non-finite values"

    # Parameters must remain on CUDA throughout training
    for name, p in merger.named_parameters():
        assert p.device.type == 'cuda', \
            f"Parameter '{name}' moved off CUDA during training"


# ---------------------------------------------------------------------------
# End-to-end CLI GPU tests
# ---------------------------------------------------------------------------

niter = 10


@_xfail_no_gpu
@pytest.mark.parametrize('mode', ['mono', 'poly'])
def test_run_careless_on_gpu(mode, off_file, on_file):
    """run_careless should complete and produce output when a GPU is available."""
    from tempfile import TemporaryDirectory
    from os.path import exists
    from careless.careless import run_careless
    from careless.parser import parser

    with TemporaryDirectory() as td:
        out     = td + '/out'
        command = (
            f"{mode} --iterations={niter} dHKL,image_id"
            f" {off_file} {on_file} {out}"
        )
        args = parser.parse_args(command.split())

        # Verify the intended device is CUDA before running
        assert torch.cuda.is_available()
        assert args.disable_gpu is False

        run_careless(args)

        out_file = out + '_0.mtz'
        assert exists(out_file), f"Output file {out_file} was not created"

        # Verify GPU memory was actually used (non-zero allocation after run)
        assert torch.cuda.max_memory_allocated(0) > 0, \
            "No CUDA memory was allocated — model may not have run on GPU"
