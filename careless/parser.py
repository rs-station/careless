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
        environ['TF_CPP_MIN_LOG_LEVEL'] = str(parser.tf_log_level)

        import tensorflow as tf
        np.random.seed(parser.seed)
        tf.random.set_seed(parser.seed)

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
        for inFN in parser.mtzinput:
            if not exists(inFN):
                self.error(f"Unmerged Mtz file {inFN} does not exist")

        if parser.prior_mtz:
            if not exists(parser.prior_mtz):
                self.error(f"Prior Mtz file {parser.prior_mtz} does not exist")

    def _validate_priors(self, parser):
        if parser.studentt_prior_dof:
            if not parser.prior_mtz:
                print(parser.studentt_prior_dof)
                self.error("--studentt-prior-dof requires --prior-mtz")
        if parser.laplace_prior:
            if not parser.prior_mtz:
                print(parser.studentt_prior_dof)
                self.error("--laplace-prior requires --prior-mtz")
        if parser.normal_prior:
            if not parser.prior_mtz:
                print(parser.studentt_prior_dof)
                self.error("--normal-prior requires --prior-mtz")

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

from careless.args.exclusive import groups
for group in groups:
    mono_group = mono.add_mutually_exclusive_group()
    poly_group = poly.add_mutually_exclusive_group()
    for args,kwargs in group:
        mono_group.add_argument(*args, **kwargs)
        poly_group.add_argument(*args, **kwargs)

if __name__=="__main__":
    #This makes debugging without running the full script easy
    parser=parser.parse_args()
    from IPython import embed
    embed(colors="Linux")
