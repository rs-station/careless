import argparse

class TfSettingsMixin():
    """
    This will automagically set tensorflow environment variables when parse_args is called.
    """
    def parse_args(self, *args, **kwargs):
        parser = super().parse_args(*args, **kwargs)

        from os import environ
        #Suppress most tensorflow output
        environ['TF_CPP_MIN_LOG_LEVEL'] = str(parser.tf_log_level)

        #Disable the GPU if requested. This can be useful for training multiple models at the same time
        if parser.disable_gpu:
            environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return parser

class ArgumentParser(TfSettingsMixin, argparse.ArgumentParser):
    pass


description = """
Scale and merge crystallographic data by approximate inference.
"""


error_models = [
    "Gaussian", 
    "StudentT", 
    "Laplace",
]

#Specific arguments for laue version with harmonic deconvolution
laue_arguments = {
    ("-l", "--wavelength-range") : {
        'help' : f"Minimum and maximum wavelength for harmonic deconvolution in Ångstroms. If this is not supplied, Harmonics will be predicted out to the minimum and maximum wavelengths recorded in the mtz.", 
        'type' : float, 
        'default': (None, None), 
        'nargs' : 2,
        #'dest': ['lambda_min', 'lambda_max'], 
    },

    ("-w", "--wavelength-key") : {
        'help' : f"Mtz column name corresponding to the reflections' peak wavelength.", 
        'type' : str, 
        'default': 'Wavelength', 
        'nargs' : 1,
        'dest': 'wavelength_key', 
    },
}

#Specific arguments for monochromatic version without harmonic deconvolution
mono_arguments = {
}

arguments = {
    ("reflection_filename" ,) : {
        'help' : "Mtz filename(s).", 
        'type' : str, 
        'nargs':'+',
        'default' : None,
    },

    ("output_mtz", ) : {
        'help': "Ouput merged reflection filename", 
        'type': str,
    },

    ("-c", "--isigi-cutoff") : {
        'help': "Minimum I over Sigma(I) for included reflections. Default is to include all reflections", 
        'type': float, 
        'default': None,
    },

    ("-e", "--error-model") : {
        'help' : f"Avalailable error models are {', '.join(error_models)}", 
        'type' : str, 
        'default' : 'Gaussian', 
    },

    ("-s", "--space-group-number") : {
        'help' : f"What space group number to merge in. By default use the Mtz spacegroup.", 
        'type' : int, 
        'default' : None,
    },

    ("-d", "--dmin") : {
        'help' : f"Maximum resolution in Ångstroms. If this is not supplied, reflections will be merged out to the highest resolution reflection present in the input.", 
        'type' : float, 
        'default': None, 
    },

    ("--seed", ) : {
        'help' : f"Random number seed for consistent half dataset generation.", 
        'type' : int, 
        'default' : 1234, 
    },

    ("--iterations", ) : {
        'help': "Number of gradient steps to take.", 
        'type': int, 
        'default': 2000, 
    },

    ("--disable-gpu", ) : {
        'help': "Disable GPU for high memory models.", 
        'type': bool, 
        'default': False, 
    },

    ("--learning-rate", ) : {
        'help': "Adam learning rate.", 
        'type': float, 
        'default': 0.1, 
    },

    ("--studentt-dof", ) : {
        'help': "Degrees of freedom for student t error model.", 
        'type': float, 
        'default': None, 
    },

    ("--mc-iterations", ) : {
        'help': "This is the number of samples to take per gradient step with default = 1.", 
        'type': int, 
        'default': 1, 
    },

    ("--merge-files", ) : {
        'help': "Use this flag to merge all the supplied Mtzs together. Otherwise the different files will be scaled together but merged separately.", 
        'type': bool, 
        'default': False, 
    },

    ("--equate-batches", ) : {
        'help': "Should batch IDs be treated the same between experiments? You might want to set this to true if, for instance you are scaling a time resolved experiment with multiple images per orientation. The default is False.", 
        'type': bool, 
        'default': False, 
    },

    ("--metadata-keys", ) : {
        'help': "Mtz column labels of metadata. If this this not supplied, all suitable columns will be used. By default Intensity, Stddev, and HKL dtypes will not be included.", 
        'type': str, 
        'nargs' : "+",
        'default': None,
    },

    ("--tf-log-level", ) : {
        'help': "Change the TensorFlow log level by setting the 'TF_CPP_MIN_LOG_LEVEL' environment variable. The default is '3' which is quiet.", 
        'type': int, 
        'nargs' : 1,
        'default': 3,
    },

    ("--weights", ) : {
        'help': "Use a weighted log likelihood term.", 
        'type': bool, 
        'nargs' : 1,
        'default': False,
    },
    
}

mono_parser = ArgumentParser()
laue_parser = ArgumentParser()

for args, kwargs in arguments.items():
    laue_parser.add_argument(*args, **kwargs)
    mono_parser.add_argument(*args, **kwargs)

for args, kwargs in laue_arguments.items():
    laue_parser.add_argument(*args, **kwargs)

for args, kwargs in mono_arguments.items():
    mono_parser.add_argument(*args, **kwargs)
