name = "Laue"
description = None

args_and_kwargs = (
    (("-l", "--wavelength-range"), {
        "help":"Minimum and maximum wavelength for harmonic deconvolution in Ångstroms. "
               "If this is not supplied, Harmonics will be predicted out to the minimum and "
               "maximum wavelengths recorded in the mtz.", 
        "type": float, 
        "default": None,
        "nargs":2, 
        "metavar": ('lambda_min', 'lambda_max'),
    }),

    (("-w", "--wavelength-key"),  {
        "help":f"Mtz column name corresponding to the reflections' peak wavelength.", 
        "type":str, 
        "default":'Wavelength',
    }),
)
