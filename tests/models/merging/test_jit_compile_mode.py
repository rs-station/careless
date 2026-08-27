"""
Tests for --jit-compile-mode plumbing.

These cover the argument handling only; the compiled numerics are covered by the
end-to-end equivalence measurements recorded in doc/performance/README.md, which
need a GPU and several minutes of compilation.
"""
import pytest

from careless.args.tf_options import (
    CUDA_GRAPH_MODES,
    JIT_COMPILE_MODES,
    args_and_kwargs,
)
from careless.models.merging.variational import VariationalMergingModel


def _kwargs_for(flag):
    for args, kwargs in args_and_kwargs:
        if flag in args:
            return kwargs
    raise AssertionError(f"{flag} is not defined in tf_options")


def test_jit_compile_mode_defaults_to_the_fastest_measured_mode():
    kwargs = _kwargs_for("--jit-compile-mode")
    assert kwargs["default"] == "max-autotune-no-cudagraphs"
    assert tuple(kwargs["choices"]) == JIT_COMPILE_MODES


@pytest.fixture
def dummy_mtz(tmp_path):
    """The parser only checks that the reflection file exists and ends in .mtz."""
    path = tmp_path / "in.mtz"
    path.touch()
    return str(path)


def test_parser_accepts_every_mode_and_rejects_others(dummy_mtz, tmp_path):
    from careless.parser import parser

    tail = ["dHKL", dummy_mtz, str(tmp_path / "out")]
    for mode in JIT_COMPILE_MODES:
        args = parser.parse_args(["mono", f"--jit-compile-mode={mode}"] + tail)
        assert args.jit_compile_mode == mode

    # and the default is applied when the flag is absent
    assert parser.parse_args(["mono"] + tail).jit_compile_mode == (
        "max-autotune-no-cudagraphs"
    )

    with pytest.raises(SystemExit):
        parser.parse_args(["mono", "--jit-compile-mode=nonsense"] + tail)


@pytest.mark.parametrize("mode", JIT_COMPILE_MODES)
def test_compile_kwargs_passes_the_mode_through(mode):
    kwargs = VariationalMergingModel._compile_kwargs(mode, reduce_retracing=False)
    assert kwargs["dynamic"] is False
    if mode == "default":
        # torch.compile has no "default" mode string; omitting it *is* the default.
        assert "mode" not in kwargs
    else:
        assert kwargs["mode"] == mode


def test_compile_kwargs_honours_reduce_retracing():
    kwargs = VariationalMergingModel._compile_kwargs(
        "max-autotune-no-cudagraphs", reduce_retracing=True
    )
    assert kwargs["dynamic"] is True


@pytest.mark.parametrize("mode", sorted(CUDA_GRAPH_MODES))
def test_cuda_graph_modes_refuse_dynamic_shapes(mode):
    """The combination segfaults, so it must be rejected before torch sees it."""
    with pytest.raises(ValueError, match="segfault"):
        VariationalMergingModel._compile_kwargs(mode, reduce_retracing=True)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown jit_compile_mode"):
        VariationalMergingModel._compile_kwargs("turbo", reduce_retracing=False)
