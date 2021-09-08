name = "Data Interpretation"
description = None



args_and_kwargs = (
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
