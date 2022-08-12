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
        if parser.tf_debug:
            # This is very noisy
            environ['TF_CPP_MIN_LOG_LEVEL'] = "1"
        else:
            # This is very quiet
            environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

        import tensorflow as tf
        np.random.seed(parser.seed)
        tf.random.set_seed(parser.seed)

        #Disable the GPU if requested. This can be useful for training multiple models at the same time
        if parser.disable_gpu:
            tf.config.set_visible_devices([], 'GPU')
        #For multi-gpu machines allocate only the zeroth GPU.
        else:
            #Set active GPU
            gpus = tf.config.experimental.list_physical_devices('GPU')
            gpu_id = parser.gpu_id

            try:
                gpu = gpus.pop(gpu_id)
                if not parser.disable_memory_growth:
                    tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.experimental.set_visible_devices(gpu, 'GPU')
            except:
                tf.config.experimental.set_visible_devices([], 'GPU')

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
        return parser

import re
import textwrap
class CustomFormatter(argparse.HelpFormatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._whitespace_matcher=re.compile("\n(?!\n)")

    def _fill_text(self, text, width, indent):
        #First replace single newlines with a space
        text = re.sub(r'(?!>\n)\n(?!\n)', '', text)
        return textwrap.fill(text, width, initial_indent=indent, subsequent_indent=indent, replace_whitespace=False, drop_whitespace=False)

description = """
Scale and merge crystallographic data by \n\n\n approximate inference.
"""
parser = CustomParser(description=description, formatter_class=CustomFormatter)

# Add --version argument
import careless
parser.add_argument("--version", action="version", version=f"careless {careless.__version__}")

subs = parser.add_subparsers(title="Experiment Type", required=True, dest="type")
mono_sub = subs.add_parser("mono", help="Process monochromatic diffraction data.", formatter_class=CustomFormatter)
poly_sub = subs.add_parser("poly", help="Process polychromatic, 'Laue', diffraction data.", formatter_class=CustomFormatter)

from careless.args import required,poly,groups

for args,kwargs in required.args_and_kwargs:
    mono_sub.add_argument(*args, **kwargs)
    poly_sub.add_argument(*args, **kwargs)

for args,kwargs in poly.args_and_kwargs:
    poly_sub.add_argument(*args, **kwargs)

for group in groups:
    if group.name is not None and group.description is not None:
        mono_group = mono_sub.add_argument_group(group.name, group.description)
        poly_group = poly_sub.add_argument_group(group.name, group.description)
    elif group.name is not None:
        mono_group = mono_sub.add_argument_group(group.name)
        poly_group = poly_sub.add_argument_group(group.name)
    else:
        mono_group = mono_sub
        poly_group = poly_sub
    for args,kwargs in group.args_and_kwargs:
        mono_group.add_argument(*args, **kwargs)
        poly_group.add_argument(*args, **kwargs)

