name = "Data Filtration"
description = None

args_and_kwargs = (
    # Required args
    (("-c", "--isigi-cutoff"), {
        "help":"Minimum I over Sigma(I) for included reflections. Default is to include all reflections", 
        "type":float, 
        "default":None,
    }),

    (("-d", "--dmin"), {
        "help":f"Maximum resolution in Ångstroms. If this is not supplied," 
                "reflections will be merged out to the highest resolution reflection present in the input.", 
        "type":float, 
        "default":None,
    }),
)
