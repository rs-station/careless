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
)