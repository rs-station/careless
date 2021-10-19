name = "Scaling Model"
description = """
Options related to the neural network scaling model used for merging. 
"""


args_and_kwargs = (
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

    (("--disable-image-scales",), {
        "help": "Do not learn a local scale param for each image.",
        "action": "store_false",
        "dest" : "use_image_scales",
        "default": True,
    }),
)
