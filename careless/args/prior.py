name = "Prior"
description = """
Options related to the prior distribution applied to the structure factors during merging.
"""

args_and_kwargs = (
    (("--wilson-prior-b",), {
        "help": "This flag enables learning reflections on a particular Wilson scale. "
                "By default, the Wilson prior is flat across resolution bins. ",
        "type": float, 
        "default": None,
    }),
)

