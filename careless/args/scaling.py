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

    (("--image-layers",), {
        "help": "Add additional layers with local image-specific parameters.",
        "type":int,
        "default": 0,
    }),


    (("--disable-image-scales",), {
        "help": "Do not learn a local scale param for each image.",
        "action": "store_false",
        "dest" : "use_image_scales",
        "default": True,
    }),

    (("--scale-bijector",), {
        "help": "What function to use to ensure positivity of the standard deviation of scales. ",
        "type": str,
        "default": "exp",
        "choices" : ["exp", "softplus"],
    }),
    (("--spectral-file",), {
        "help": "Path to a two-column whitespace-delimited text file ('wavelength scale') representing the incident flux spectrum."
                "Disables neural network scaling. Used for harmonic deconvolution with a known spectrum."
                "See docs/spectral_scaling.md",
        "type": str,
        "default": None,
    }),
    (("--trainable-spectral-scale",), {
        "help": "If set, multiplies the tabulated spectrum by a single learnable global scalar. "
                "Allows the overall magnitude to float while keeping the spectral shape fixed.",
        "action": "store_true",
        "default": False,
    }),
    (("--spectral-grid-points",), {
        "help": "Number of points to use for the interpolated spectral lookup table. "
                "Higher values provide more accuracy but consume more memory. "
                "Default is 10,000.",
        "type": int,
        "default": 10000,
    }),
    (("--lorentz-correction",), {
        "help": "Apply the Laue Lorentz correction factor (L ~ lambda^4 / sin^2(theta)) to the scales. "
                "Requires using the tabulated spectral scaler.",
        "action": "store_true",
        "default": False,
    }),
)
