name = "Positional Encoding"
description = """
The positional encoding is a data augmentation strategy first developed in the natural language processing literature. 
It has gained widespread adoption as a simple and effective strategy to improve the performance of neural networks in 
modelling spatially resolved functions (https://arxiv.org/abs/2003.08934, https://arxiv.org/abs/1909.05215). Careless
provides the ability to apply the positional encoding to a subset of reflection metadata using the 
`--positional-encoding-keys` flag. \n

  Examples\n
  --------\n
    careless mono --positional-encoding-keys="XDET,YDET" "Hobs,Kobs,Lobs,BATCH" input.mtz out\n
"""

args_and_kwargs = (

    (("--positional-encoding-keys", ), {
        "help":"If the `--positional-encoding-frequencies` flag is set to an integer > 1, this parameter enables encoding a specific subset of"
               'of mtz columns. Supply a comma separated string of metadata keys (ie "XDET,YDET"), and these keys will be encoded separately and '
               'appended to the rest of the metadata. ', 
        "type" : str,
        "default" : None,
    }),

    (("--positional-encoding-frequencies", "-L"), {
        "help":"Number of positional encoding frequencies to apply to metadata. "
               "If you use this option, it should be paired with 'mlp-width=' in order to prevent the model from using too much memory."
               "By default all metadata columns will be encoded using the same formula. To encode a subset of the columns, please see"
               "the `--positional-encoding-keys` parameter",
        "type" : int,
        "default" : 4,
    }),

)
