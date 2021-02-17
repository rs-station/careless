#This file contains arguments that should be common to all running modes of careless

args_and_kwargs = (
    (("metadata_keys", ),  {
        "help":"Metadata keys for scaling. This is expected to be a comma delimitted string",
        "type":str, 
    }),

    (("--embed", ),  {
        "help":"Drop to an IPython shell at the end of optimization to play around.",
        "action" : "store_true",
        "default" : False,
    }),

    (("--multiplicity-weighted-elbo", ), {
        "help":"Reweight the kl_div term by multiplicity. This may perform better in some cases.",
        "action" : "store_true",
        "default" : False,
    }),

    (("--use-nadam", ), {
        "help":"Instead of using the Adam optimizer, use the Nadam optimizer which has Nesterov momentum.",
        "action" : "store_true",
        "default" : False,
    }),

    (("--use-weights", ),  {
        "help":"Use a weighted likelihood function.",
        "action" : "store_true",
        "default" : False,
    }),

    (("--mc-samples",), {
        "help":"This is the number of samples to take per gradient step with default 1. " ,
        "type": int, 
        "default" : 1,
    }),

    (("--skip-xval", ),  {
        "help":"Bypass merging half data sets.",
        "action" : "store_true",
        "default" : False,
    }),


    (("--image-scale-key", ),  {
        "help":"Key to use for per image scaling. ",
        "type":str, 
        "default" : None,
    }),

    (("--image-scale-prior", ),  {
        "help":"Fractional scale of the prior (normal) distribution on image scales. ",
        "type":float, 
        "default" : None,
    }),

    (("--image-id-key", ),  {
        "help":"The name of the key indicating image number for each data set. "
               "If no key is given, careless will use the first key with the BATCH dtype.",
        "type":str, 
        "default" : None,
    }),

    (("--intensity-key", ),  {
        "help":"What key to use for reflection intensities. "
               "If no key is given, careless will use the first key with the intensity dtype.",
        "type":str, 
        "default" : None,
    }),

    (("--folded-normal-surrogate", ), {
        "help":"Use a folded normal (woolfson) distribution as the surrogate posterio instead of truncated normal",
        "action":"store_true",
        "default": False,
    }),

    (("--rice-woolfson-surrogate", ), {
        "help":"Use a hybrid rice/woolfson distribution as the surrogate posterio instead of truncated normal",
        "action":"store_true",
        "default": False,
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

# This is not supported yet
#    (("-s", "--space-group"), {
#        "help":f"Spacegroup number or symbol to merge in. By default use the Mtz spacegroup.", 
#        "type":str, 
#        "default":None,
#    }),

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
        "help":"Adam learning rate. The default is 0.001", 
        "type":float, 
        "default":0.001,
    }),

    (("--clip-value",), {
        "help":"Maximum gradient absolute value.", 
        "type": float, 
        "default": None,
    }),

    (("--beta-1",), {
        "help":"Adam beta_1 param.", 
        "type":float, 
        "default":0.9,
    }),

    (("--beta-2",), {
        "help":"Adam beta_2 param.", 
        "type":float, 
        "default":0.99,
    }),

    (("--separate-files",), {
        "help":"Use this flag to produce a separate output for each input mtz." 
               "In this mode, the data will be 'scaled' together and 'merged' separately." 
               "The default is to merge all the files into a single output.", 
        "action" : "store_true",
        "default": False,
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

    (("--tf-log-level",), {
        "help": "Change the TensorFlow log verbosity by setting the "
                "TF_CPP_MIN_LOG_LEVEL environment variable. "
                "The default is 3 which is quiet.", 
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

    (("--sequential-layers",), {
        "help": "The number of sequential dense neural network layers in the scaling model.",
        "type":int,
        "default":20,
    }),

    (("--studentt-scale",), {
        "help": "Scale parameter for variational student t likelihood.",
        "type": float,
        "default": None,
    }),

)
