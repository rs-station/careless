name = None
description = None

args_and_kwargs = (
    # Required args
    (("metadata_keys", ),  {
        "help":"Metadata keys for scaling. This is expected to be a comma delimitted string",
        "type":str, 
    }),

    (("reflection_files", ), { 
        "metavar":"reflections.{mtz,stream}", 
        "help":"Mtz or stream file(s) containing unmerged reflection observations. If you are supplying"
               " stream files, you must also use the --spacegroups option to supply the symmetry for "
               "merging.", 
        "type":str, 
        "nargs":'+',
    }),

    (("output_base", ), {
        "metavar":"out", 
        "help":"Output filename base.", 
        "type":str,
    }),
)

