"""
Compute CCpred from careless output.

"""
import argparse
import numpy as np
import reciprocalspaceship as rs
import gemmi

import matplotlib.pyplot as plt
import seaborn as sns


class BaseParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        super().__init__(
            formatter_class=argparse.RawTextHelpFormatter, 
            **kwargs
        )

        self.add_argument(
            "-s",
            "--show",
            action="store_true",
            help="Make a plot of the results and display it using matplotlib.",
        )

        self.add_argument(
            "-i",
            "--image",
            type=str,
            default=None,
            help="Make a plot of the results and save it to this filename. "
                 "The filetype will be determined from the filename. "
                 "Any filetype supported by your matplotlib version will be available.",
        )

        self.add_argument(
            "-o",
            "--output",
            type=str,
            default=None,
            help="Optionally save results to this file in csv format instead of printing "
                 "them to the terminal.",
        )



