name = "Scaling Model"
description = """
Options related to the neural network scaling model used for merging. 
"""


args_and_kwargs = (
    (("--scale-file",), {
        "help": "Initialize the scale model weights from the ouput of a previous run. This argument should be a string beginning with the "
                "base filename used in the previous run and ending in _scale.  For instance, if the previous run "
                "was called with `careless mono [...] merge/hewl`, the appropriate file name would be merge/hewl_scale. ",
        "type": str,
        "default": None,
    }),

    (("--freeze-scales",), {
        "help": "Do not optimize the scale model weights.",
        "action": "store_true"
    }),

    (("--mlp-layers",), {
        "help": "The number of dense neural network layers in the scaling model. The default is 20 layers.",
        "type":int,
        "default":20,
    }),

    (("--mlp-width",), {
        "help": "The width of the hidden layers of the neural net. The default is 10.",
        "type": int,
        "default": 10,
    }),

)
