#This file contains arguments that should be common to all running modes of careless

args_and_kwargs = (
    (("metadata_keys", ),  {
        "help":"Metadata keys for scaling. This is expected to be a comma delimitted string",
        "type":str, 
    }),

    (("reflection_files", ), { 
        "metavar":"reflections.{mtz,stream}", 
        "help":"Mtz or stream file(s) containing unmerged reflection observations.", 
        "type":str, 
        "nargs":'+',
    }),

    (("output_base", ), {
        "metavar":"out", 
        "help":"Output filename base.", 
        "type":str,
    }),

    (("--embed", ),  {
        "help":"Drop to an IPython shell at the end of optimization to play around.",
        "action" : "store_true",
        "default" : False,
    }),

    (("--positional-encoding-frequencies", "-L"), {
        "help":"Number of positional encoding frequencies to apply to metadata. The default is 1 which corresponds to no encoding."
               "If you use this option, it should be paired with 'mlp-width=' in order to prevent the model from using too much memory."
               "By default all metadata columns will be encoded using the same formula. To encode a subset of the columns, please see"
               "the `--positional-encoding-keys` parameter",
        "type" : int,
        "default" : 1,
    }),

    (("--positional-encoding-keys", ), {
        "help":"If the `--positional-encoding-frequencies` flag is set to an integer > 1, this parameter enables encoding a specific subset of"
               'of mtz columns. Supply a comma separated string of metadata keys (ie "XDET,YDET"), and these keys will be encoded separately and '
               'appended to the rest of the metadata. ', 
        "type" : str,
        "default" : None,
    }),

    (("--mc-samples",), {
        "help":"This is the number of samples to take per gradient step with default 1. " ,
        "type": int, 
        "default" : 1,
    }),

    (("--test-fraction", ),  {
        "help":"Perform cross validation on a held out fraction of data.",
        "type" : float,
        "default" : None,
    }),

    (("--image-key", ),  {
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

    (("--anomalous",), { 
        "help":f"If this flag is supplied, Friedel mates will be kept separate.", 
        "action":'store_true', 
        "default":False,
    }),

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

    (("--mlp-layers",), {
        "help": "The number of dense neural network layers in the scaling model.",
        "type":int,
        "default":20,
    }),

    (("--mlp-width",), {
        "help": "Use a different width for the hidden layers of the neural net than the width of the metadata array.",
        "type": int,
        "default": None,
    }),

    (("--wilson-prior-b",), {
        "help":"Experimental Feature: "
               "This flag enables learning reflections on a particular Wilson scale. "
               "By default, the Wilson prior is flat across resolution bins. ",
        "type": float, 
        "default": None,
    }),

    (("--studentt-likelihood-dof",), { 
        "help":"Degrees of freedom for student t likelihood function.",
        "type":float, 
        "metavar":'DOF', 
        "default":None,
    }),

)
