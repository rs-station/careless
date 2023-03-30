"""
Compute completeness from careless output.
"""
import argparse
import matplotlib.pyplot as plt
import reciprocalspaceship as rs
import seaborn as sns
import numpy as np


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(
            formatter_class=argparse.RawTextHelpFormatter, 
            description=__doc__
        )

        # Required arguments
        self.add_argument(
            "mtz",
            help="MTZ containing merged data from careless",
        )

        self.add_argument(
            "-b",
            "--bins",
            default=10,
            type=int,
            help=("Number of resolution bins to use, the default is 10."),
        )

        self.add_argument(
            "-o",
            "--output",
            type=str,
            default=None,
            help=("Optionally save completeness values to this file in csv format."),
        )

def run_analysis(args, show=True):
    ds = rs.read_mtz(args.mtz)
    results = rs.stats.compute_completeness(ds, bins=args.bins)

    if args.output is not None:
        results.to_csv(args.output)

    #Move overall to the beginning
    results = results.iloc[np.roll(np.arange(len(results)), 1)]

    ax = sns.lineplot(
        data=results.reset_index().melt('index'),
        x='index',
        y='value',
        hue='variable_1',
        palette='Dark2',
    )
    plt.xticks(rotation=45, rotation_mode='anchor', ha='right')
    plt.legend(title="")
    plt.ylabel(r"Completeness")
    plt.xlabel("Resolution ($\mathrm{\AA}$)")
    plt.grid(which='both', axis='both', ls='dashdot')
    plt.tight_layout()
    if show:
        print(results.to_string())
        plt.show()


def main():
    parser = ArgumentParser().parse_args()
    run_analysis(parser, True)

