import argparse
import numpy as np 
from os.path import exists

class EnvironmentSettingsMixin(argparse.ArgumentParser):
    """
    This will automagically set tensorflow environment variables when parse_args is called.
    """
    def parse_args(self, *args, **kwargs):
        parser = super().parse_args(*args, **kwargs)

        from os import environ
        if parser.tf_dbug:
            # This is very noisy
            environ['TF_CPP_MIN_LOG_LEVEL'] = 1
        else:
            # This is very quiet
            environ['TF_CPP_MIN_LOG_LEVEL'] = 3

        import tensorflow as tf
        np.random.seed(parser.seed)
        tf.random.set_seed(parser.seed)

        #Run eagerly if requested. This is very slow but needed for very large models
        if parser.run_eagerly:
            tf.config.run_functions_eagerly(True)

        #Disable the GPU if requested. This can be useful for training multiple models at the same time
        if parser.disable_gpu:
            tf.config.set_visible_devices([], 'GPU')

        physical_devices = tf.config.list_physical_devices('GPU')
        if not parser.disable_memory_growth:
            try:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
            except:
                # Invalid device or cannot modify virtual devices once initialized.
                pass

        return parser

class CustomParser(EnvironmentSettingsMixin):
    """
    A custom ArgumentParser with parse_args overloaded in order to 
     - Set tensorflow environment variables
     - Detect conflicting arguments and raise an informative error
    """
    def _validate_input_files(self, parser):
        for inFN in parser.reflection_files:
            if not exists(inFN):
                self.error(f"Unmerged reflection file {inFN} does not exist")
            elif inFN.endswith(".mtz") or inFN.endswith(".stream"):
                continue
            self.error(
                f"Could not determine filetype for reflection file, {inFN}." 
                 "Please make sure your files end in '.mtz' or '.stream' as"
                 " appropriate."
                )

    def parse_args(self, *args, **kwargs):
        parser = super().parse_args(*args, **kwargs)
        self._validate_input_files(parser)
        self._validate_priors(parser)
        return parser


description = """
Scale and merge crystallographic data by approximate inference.
"""

parser = CustomParser(description=description)
subs = parser.add_subparsers(title="Experiment Type", required=True, dest="type")
mono = subs.add_parser("mono", help="Process monochromatic diffraction data.")
poly = subs.add_parser("poly", help="Process polychromatic, 'Laue', diffraction data.")

from careless.args.common import args_and_kwargs
for args,kwargs in args_and_kwargs:
    mono.add_argument(*args, **kwargs)
    poly.add_argument(*args, **kwargs)

from careless.args.poly import args_and_kwargs
for args,kwargs in args_and_kwargs:
    poly.add_argument(*args, **kwargs)


if __name__=="__main__":
    #This makes debugging without running the full script easy
    parser=parser.parse_args()
    from IPython import embed
    embed(colors="Linux")
