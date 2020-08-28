groups = (
    ( 
        (("--laplace-prior",), {
            "help":"Use empirical reference structure factor apmlitudes " 
                   "from an Mtz file to parameterize a Laplacian prior distribution. ",
            "action":'store_true', 
            "default":False,
        }),

        (("--normal-prior",), {
            "help":"Use empirical reference structure factor apmlitudes " 
                   "from an Mtz file to parameterize a Normal prior distribution. ",
            "action":'store_true', 
            "default":False,
        }),

        (("--studentt-prior-dof",), {
            "help":"Use empirical reference structure factor apmlitudes "
                   "from an Mtz file to parameterize a Student T prior distribution. "
                   "Must specify an mtz file and the degrees of freedom. ",
            "type": float, 
            "default": None,
        }),
    ),

    (
        (("--studentt-likelihood-dof",), { 
            "help":"Degrees of freedom for student t likelihood function.",
            "type":float, 
            "metavar":'DOF', 
            "default":None,
        }),

        (("--laplace-likelihood",), { 
            "help":"Use a Laplacian likelihood function.", 
            "default":False, 
            "action":'store_true',
        }),
    ),

    (
        (("--mc-samples",), {
            "help":"This is the number of samples to take per gradient step with default 1. " 
            "This option is incompatible with quadrature.", 
            "type": int, 
            "default" : 1,
        }),

        (("--quadrature-points",), {
                "help":"Use quadrature at this many points to estimate the expected log "
                       "likelihood. For normally distributed likelihoods, 3 is sufficient. "
                       "For all other likelihoods 10 points is recommended. "
                       "This option is incompatible with --mc-samples",
                "type": int, 
            "default" : None,
        }),
    ),
)
