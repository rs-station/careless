name = "Scaling Model"
description = """
Options related to the neural network scaling model used for merging. 
"""


args_and_kwargs = (
    (("--scale-file",), {
        "help": "Load scaling model weights from previous careless output. This should be a string beginning with the [output_base]"
        "from a previous run (ie 'merge/hewl_scale').",
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
        "help": "The width of the hidden layers of the neural net. This defaults to the dimensionality of the metadata array.",
        "type": int,
        "default": None,
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
)
