name=None
description=None

args_and_kwargs = (
    (("--embed", ),  {
        "help":"Drop to an IPython shell at the end of optimization to inpsect variables.",
        "action" : "store_true",
        "default" : False,
    }),

    (("--mc-samples",), {
        "help":"This is the number of samples to take per gradient step with default 1. " ,
        "type": int, 
        "default" : 1,
    }),

    (("--structure-factor-file",), {
        "help": "Initialize the structure factors from the ouput of a previous run. This argument should be a string beginning with the "
                "base filename used in the previous run and ending in _structure_factor.  For instance, if the previous run "
                "was called with `careless mono [...] merge/hewl`, the appropriate filename to use would be merge/hewl_structure_factor. ",
        "type": str, 
        "default" : None,
    }),


    (("--freeze-structure-factors",), {
        "help": "Do not optimize the structure factors.",
        "action": "store_true"
    }),

    (("--structure-factor-init-scale",), {
        "help":"A floating point number usually between 0 and 1. The width of the initial structure factor distribution is this times" 
               "the standard deviation of the prior distribution. The default is 1.0. ",
        "type": float, 
        "default" : 1.0,
    }),

    (("--epsilon",), {
        "help":"A small constant added to the scale parameters of variational distributions  for numerical stability. The default is 1e-7." ,
        "type": float, 
        "default" : 1e-7,
    }),

    (("--disable-metadata-standardization",), {
        "help": "By default careless will convert metadata to z-scores. This flag disables that behavior. In general, unstandardized metadata will lead to unstable optimization. However, this flag might be useful if the user wants to use their own normalization scheme." ,
        "action": "store_false", 
        "dest" : "standardize_metadata", 
    }),

    (("--disable-progress-bar", ),  {
        "help":"Disable the progress bar. This is helpful if you're logging stderr.",
        "action" : "store_true",
        "default" : False,
    }),
)
