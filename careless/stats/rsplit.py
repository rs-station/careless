"""
Compute Rsplit from careless output.
"""
import argparse
import matplotlib.pyplot as plt
import reciprocalspaceship as rs
import seaborn as sns
from scipy.optimize import minimize
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
            nargs="+",
            help="MTZs containing crossvalidation data from careless",
        )

        self.add_argument(
            "-b",
            "--bins",
            default=10,
            type=int,
            help=("Number of resolution bins to use, the default is 10."),
        )

        self.add_argument(
            "--use-intensities",
            action="store_true",
            help=("Optionally use intensities instead of structure factors to facilitate comparisons with other softwares."),
        )

        self.add_argument(
            "-o",
            "--output",
            type=str,
            default=None,
            help=("Optionally save Rsplit values to this file in csv format."),
        )


def make_halves_cchalf(mtz, bins=10):
    """Construct half-datasets for computing Rsplit"""

    half1 = mtz.loc[mtz.half == 0].copy()
    half2 = mtz.loc[mtz.half == 1].copy()

    # Support anomalous
    if "F(+)" in half1.columns:
        half1 = half1.stack_anomalous()
        half2 = half2.stack_anomalous()

    # Using the definition of variance
    half1["I"] = half1["F"] * half1["F"] + half1["SigF"] * half1["SigF"]
    half2["I"] = half2["F"] * half2["F"] + half2["SigF"] * half2["SigF"]

    temp = half1[["I", "F", "repeat"]].merge(
        half2[["I", "F", "repeat"]], on=["H", "K", "L", "repeat"], suffixes=("1", "2")
    )
    temp, labels = temp.assign_resolution_bins(bins)

    return temp, labels


def rsplit(x, y):
    def rfunc(k):
        return np.sum(np.abs(x - k * y)) / np.sum(x + k * y)

    p = minimize(rfunc, 1.)
    return np.sqrt(2) * p.fun 

def analyze_cchalf_mtz(mtzpath, bins=10, return_labels=True, keys=("F1", "F2")):
    """Compute Rsplit from 2-fold cross-validation"""

    mtz = rs.read_mtz(mtzpath)

    # Error handling -- make sure MTZ file is appropriate
    if "half" not in mtz.columns:
        raise ValueError("Please provide MTZs from careless crossvalidation")

    m, labels = make_halves_cchalf(mtz, bins)

    grouper = m.groupby(["bin", "repeat"])[list(keys)]
    result = (
        grouper.corr(method=rsplit).unstack()[keys].to_frame().reset_index()
    )

    if return_labels:
        return result, labels
    else:
        return result


def run_analysis(args, show=True):
    results = []
    labels = None

    if args.use_intensities:
        keys = ("I1", "I2")
    else:
        keys = ("F1", "F2")

    for m in args.mtz:
        result = analyze_cchalf_mtz(m, bins=args.bins, keys=keys)
        if result is None:
            continue
        else:
            result[0]["filename"] = m
            results.append(result[0])
            labels = result[1]

    results = rs.concat(results, check_isomorphous=False)
    results = results.reset_index(drop=True)
    results["Rsplit"] = results[keys]
    results.drop(columns=[keys], inplace=True)


    for k in ('bin', 'repeat'):
        results[k] = results[k].to_numpy('int32')

    if args.output is not None:
        results.to_csv(args.output)
    
    sns.lineplot(
        data=results, x="bin", y="Rsplit", hue="filename", palette="Dark2"
    )
    plt.xticks(range(args.bins), labels, rotation=45, ha="right", rotation_mode="anchor")
    plt.ylabel(r"$\mathrm{R_{split}}$ ")
    plt.xlabel("Resolution ($\mathrm{\AA}$)")
    plt.grid(which='both', axis='both', ls='dashdot')
    plt.tight_layout()
    if show:
        print(results.to_string())
        plt.show()


def main():
    parser = ArgumentParser().parse_args()
    run_analysis(parser, True)

