name = "Data Interpretation"
description = None



args_and_kwargs = (
    (("--spacegroups", ),  {
        "help":"The spacegroup(s) to use for merging. You may either supply a single spacegroup "
               "which will be used for every input reflection file or a comma-separated list of "
               "spacegroups with the same length as the number of reflection files. For example "
               '--spacegroups="P 21 21 21" or --spacegroups="P 21 21 21,P 1 21 1"',
        "type":str, 
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

    (("--anomalous",), { 
        "help":f"If this flag is supplied, Friedel mates will be kept separate.", 
        "action":'store_true', 
        "default":False,
    }),

    (("--separate-files",), {
        "help":"Use this flag to produce a separate output for each input mtz." 
               "In this mode, the data will be 'scaled' together and 'merged' separately." 
               "The default is to merge all the files into a single output.", 
        "action" : "store_true",
        "default": False,
    }),
)
