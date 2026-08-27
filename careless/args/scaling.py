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

    (("--row-major-image-layers",), {
        "help": "Run the per-image scale layers in reflection order, gathering one "
                "weight matrix per reflection. This is the original behaviour; it uses "
                "memory quadratic in --mlp-width and fails outright once "
                "n_reflections * mlp_width**2 exceeds 2**31. The default packs "
                "reflections into a padded image-major layout instead, which is "
                "numerically identical but needs no per-reflection gather.",
        "action": "store_false",
        "dest": "image_major_scaling",
        "default": True,
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
)
