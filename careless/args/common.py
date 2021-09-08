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
)
