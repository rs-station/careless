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
        "help":"Weights file from a previous careless run to initialize the structure factors. " 
               "This should be a string beginning with the [output_base] from a previous run (ie 'merge/hewl_structure_factor').",
        "type": str, 
        "default" : None,
    }),

    (("--freeze-structure-factors",), {
        "help": "Do not optimize the structure factors.",
        "action": "store_true"
    }),

    (("--kl-weight",), {
        "help": "Optionally set an explicit weight for the kl divergence. This will change the reduction of likelihood and kl_divergence"
                " from summation to means. ",
        "type": float,
        "default" : None,
    })
)
