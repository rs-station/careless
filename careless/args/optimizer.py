name = "Optimizer Parameters"
description = None

args_and_kwargs = (
    (("--iterations",), {
        "help":"Number of gradient steps to take.", 
        "type":int, 
        "default":10000,
    }),

    (("--learning-rate",), {
        "help":"Adam learning rate. The default is 0.001", 
        "type":float, 
        "default":0.001,
    }),

    (("--beta-1",), {
        "help":"Adam beta_1 param. The default is 0.9", 
        "type":float, 
        "default":0.9,
    }),

    (("--beta-2",), {
        "help":"Adam beta_2 param. The default is 0.99", 
        "type":float, 
        "default":0.99,
    }),

    (("--clipnorm",), {
        "help":"Optionally clip the norm of the gradient of each weight to be no larger than this value.", 
        "type": float, 
        "default": None,
    }),

    (("--clipvalue",), {
        "help":"Optionally clip the gradients to be no larger than this value.", 
        "type": float, 
        "default": None,
    }),

    (("--global-clipnorm",), {
        "help":"Optionally clip the norm of all the gradients to be no larger than this value.",
        "type": float,
        "default": None,
    }),

    (("--adam-epsilon",), {
        "help": "Epsilon parameter for the Adam optimizer. Default is 1e-7.",
        "type": float,
        "default": 1e-7,
    }),

    (("--disable-gradient-nan-filter",), {
        "help": "Disable per-element filtering of non-finite gradients. "
                "By default, NaN/Inf values in gradients are replaced with zero "
                "before the optimizer step (matching TF behaviour).",
        "action": "store_false",
        "dest": "filter_nan_gradients",
        "default": True,
    }),

)
