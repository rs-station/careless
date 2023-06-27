"""
Rescale careless output to match a given Wilson b-factor. 
"""
import argparse
import numpy as np
import reciprocalspaceship as rs
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from careless.stats.prior_b import estimate_b

class ArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(
            formatter_class=argparse.RawTextHelpFormatter, 
            description=__doc__
        )

        # Required arguments
        self.add_argument(
            "mtz_in",
            help="MTZs file containing unmerged data",
        )

        # Required arguments
        self.add_argument(
            "mtz_out",
            help="Output mtz file name.",
        )

        # Optional arguments
        self.add_argument(
            "-b",
            "--wilson-b",
            type=float,
            required=True,
            help="Target wilson b-factor.",
        )

def run_analysis(parser):
    ds = rs.read_mtz(parser.mtz_in)
    dHKL = ds.compute_dHKL().dHKL.to_numpy('float32')
    id2 = np.reciprocal(np.square(dHKL))
    B = parser.wilson_b
    ds['F'] = ds['F'] * np.exp(-0.25 * B * id2)
    ds['SigF'] = ds['SigF'] * np.exp(-0.25 * B * id2)

    ds['I'] = ds['I'] * np.exp(-0.5 * B * id2)
    ds['SigI'] = ds['SigI'] * np.exp(-0.5 * B * id2)

    ds.write_mtz(parser.mtz_out)


def main():
    parser = ArgumentParser().parse_args()
    run_analysis(parser)

