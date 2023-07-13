name = "Prior"
description = """
Options related to the prior distribution applied to the structure factors during merging.
"""

args_and_kwargs = (
    (("--kl-weight",), {
        "help": "Set the weight of the kl divergence term relative to the likliehood. "
                "By default, by default this is based purely on the number of reflections.",
        "type": float, 
        "default": None,
    }),

    (("--wilson-prior-b",), {
        "help": "This flag enables learning reflections on a particular Wilson scale. "
                "By default, the Wilson prior is flat across resolution bins. ",
        "type": float, 
        "default": None,
    }),

    (("--double-wilson-r",), {
        "help": "For each input mtz, designate a prior correlation coefficient with its parent. "
                "Supply one float for each file separated by commas. Supply zero for each root node."
                "for example, --double-wilson-r=0.,0.9. ",
        "type": str, 
        "default": None,
        "dest" : "dwr"
    }),

    (("--double-wilson-parents",), {
        "help": "For each input mtz, designate a parent upon which its prior is conditioned. "
                "Supply one integer for each file separated by commas. Supply None for root nodes"
                "which follow single Wilson priors. "
                "for example, --double-wilson-parents=None,0 ",
        "type": str, 
        "default": None,
        "dest" : "parents"
    }),
)

