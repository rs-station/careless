#This file contains arguments that should be common to all running modes of careless

args_and_kwargs = (
    (("metadata-keys", ),  {
        "help":"Metadata keys for scaling. This is expected to be a comma delimitted string",
        "type":str, 
    }),

    (("mtzinput", ), { 
        "metavar":"file.mtz", 
        "help":"Mtz filename(s).", 
        "type":str, 
        "nargs":'+',
    }),

    (("output_base", ), {
        "metavar":"out", 
        "help":"Output filename base.", 
        "type":str,
    }),

    (("-c", "--isigi-cutoff"), {
        "help":"Minimum I over Sigma(I) for included reflections. Default is to include all reflections", 
        "type":float, 
        "default":None,
    }),

    (("-s", "--space-group"), {
        "help":f"Spacegroup number or symbol to merge in. By default use the Mtz spacegroup.", 
        "type":str, 
        "default":None,
    }),

    (("-d", "--dmin"), {
        "help":f"Maximum resolution in Ångstroms. If this is not supplied," 
                "reflections will be merged out to the highest resolution reflection present in the input.", 
        "type":float, 
        "default":None,
    }),

    (("--anomalous",), { 
        "help":f"If this flag is supplied, Friedel mates will be kept separate.", 
        "action":'store_true', 
        "default":False,
    }),

    (("--iterations",), {
        "help":"Number of gradient steps to take.", 
        "type":int, 
        "default":2000,
    }),

    (("--learning-rate",), {
        "help":"Adam learning rate.", 
        "type":float, 
        "default":0.01,
    }),

    (("--mc-iterations",), {
        "help":"This is the number of samples to take per gradient step with default 1", 
        "type": int, 
        "default": 1,
    }),

    (("--merge-files",), {
        "help":"Use this flag to merge all the supplied Mtzs together." 
               "Otherwise the different files will be scaled together but merged separately.", 
        "action":'store_true', 
        "default":False,
    }),

    (("--disable-gpu",), {
        "help":"Disable GPU for high memory models.", 
        "action":'store_true', 
        "default":False,
    }),

    (("--tf-log-level",), {
        "help": "Change the TensorFlow log level by setting the "
                "'TF_CPP_MIN_LOG_LEVEL' environment variable. The default is '3' which is quiet.", 
        "type":int, 
        "nargs":1, 
        "default":3,
    }),

    (("--seed",), { 
        "help":f"Random number seed for consistent sampling.", 
        "type":int, 
        "default":1234, 
    }),

    (("--prior-mtz",), { 
        "help":f"Mtz containing prior structure factors and standard deviations.", 
        "type":str, 
        "default":None, 
    }),

)
