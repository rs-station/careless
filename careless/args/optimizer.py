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

)
