#: torch.compile modes accepted by --jit-compile-mode, fastest-first ordering is
#: not implied; the default below was the measured winner.
JIT_COMPILE_MODES = (
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
)

#: Modes that hand the compiled graph to CUDA graphs. Combining any of these with
#: dynamic shapes (--reduce-retracing) segfaults on torch 2.13 / triton 3.7.
CUDA_GRAPH_MODES = frozenset({"reduce-overhead", "max-autotune"})

name = "TensorFlow"
description = None

args_and_kwargs = (
    (("--run-eagerly",), {
        "help":"Running tensorflow in eager mode may be required for high memory models.", 
        "action":'store_true', 
        "default":False,
    }),

    (("--jit-compile",), {
        "help":"Compile the training step with torch.compile. This is a large speedup -- "
               "roughly 2.9x wall clock and 2.5x less peak memory on an RTX A6000 -- at "
               "the cost of a one-time compilation of up to a minute at startup. "
               "See --jit-compile-mode to choose the compiler mode.",
        "action":'store_true', 
        "default":None,
    }),

    (("--jit-compile-mode",), {
        "help":"torch.compile mode used when --jit-compile is given. 'default' compiles "
               "quickly; the max-autotune modes benchmark several kernels per operation "
               "and are much faster to run but slower to compile. The default, "
               "'max-autotune-no-cudagraphs', was the fastest and the least memory hungry "
               "of the four in a production-parameter benchmark. "
               "CUDA graphs (used by 'reduce-overhead' and 'max-autotune') buy nothing "
               "here because the sampler's accept-reject loop breaks the graph.",
        "choices":JIT_COMPILE_MODES,
        "default":"max-autotune-no-cudagraphs",
    }),

    (("--reduce-retracing",), {
        "help":"Allow dynamic shapes during compilation, so that a change in the number "
               "of reflections does not force a recompile. Disabled by default. It makes "
               "no measurable difference to careless, whose shapes are fixed for the whole "
               "run, and it cannot be combined with a CUDA-graphs --jit-compile-mode.",
        "action":'store_true', 
        "default": False,
    }),

    (("--disable-gpu",), {
        "help":"Disable GPU for high memory models.", 
        "action":'store_true', 
        "default":False,
    }),

    (("--gpu-id",), {
        "help":"Specify the physical device used for acceleration. This is an integer from"
               "0 to num accelerators - 1. The default is zero. If `--disable-gpu` is set,"
               "this option is ignored.",
        "type":int,
        "default": 0,
    }),

    (("--disable-memory-growth",), {
        "help":"Disable the experimental dynamic memory allocation.", 
        "action":'store_true', 
        "default":False,
    }),

    (("--tf-debug",), {
        "help": "Increase the TensorFlow log verbosity by setting the "
                "TF_CPP_MIN_LOG_LEVEL environment variable. ",
        "action" : 'store_true',
        "default":False,
    }),

    (("--seed",), { 
        "help":f"Random number seed for consistent sampling.", 
        "type":int, 
        "default":1234, 
    }),

)
