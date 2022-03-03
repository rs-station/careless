name = "Likelihood Options"
description = None

args_and_kwargs = (
    (("--studentt-likelihood-dof",), { 
        "help":"Degrees of freedom for student t likelihood function.",
        "type":float, 
        "metavar":'DOF', 
        "default":None,
    }),

    (("--refine-uncertainties",), { 
        "help":"Use Evans' 2011 error model from SCALA to correct uncertainties.",
        "action":'store_true', 
        "default":False,
    }),
)

