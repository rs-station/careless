name = "TensorFlow"
description = None

args_and_kwargs = (
    (("--run-eagerly",), {
        "help":"Running tensorflow in eager mode may be required for high memory models.", 
        "action":'store_true', 
        "default":False,
    }),

    (("--disable-gpu",), {
        "help":"Disable GPU for high memory models.", 
        "action":'store_true', 
        "default":False,
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
