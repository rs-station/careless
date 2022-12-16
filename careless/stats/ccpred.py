"""
Compute CCpred from careless output.

"""
import argparse
import numpy as np
import reciprocalspaceship as rs
import gemmi

import matplotlib.pyplot as plt
import seaborn as sns


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(
            formatter_class=argparse.RawTextHelpFormatter, 
            description=__doc__
        )

        # Required arguments
        self.add_argument(
            "mtzs",
            nargs="+",
            help="MTZs containing prediction data from careless",
        )

        # Optional arguments
        self.add_argument(
            "-m",
            "--method",
            default="spearman",
            choices=["spearman", "pearson"],
            help="Method for computing correlation coefficient (spearman or pearson)",
        )

        self.add_argument(
            "-b",
            "--bins",
            default=10,
            type=int,
            help="Number of resolution bins to use, the default is 10.",
        )

        self.add_argument(
            "-o",
            "--output",
            type=str,
            default=None,
            help=("Optionally save CCpred values to this file in csv format."),
        )


def compute_ccpred(
    mtzpath, overall=False, bins=10, return_labels=True, method="spearman"
):
    """Compute CCpred from cross-validation"""

    mtz = rs.read_mtz(mtzpath)

    if overall:
        labels = ['Overall']
        mtz['bin'] = -1
        grouper = mtz.groupby(["test"])[["Iobs", "Ipred"]]
    else:
        mtz, labels = mtz.assign_resolution_bins(bins)
        grouper = mtz.groupby(["bin", "test"])[["Iobs", "Ipred"]]

    result = (
        grouper.corr(method=method)
        .unstack()[("Iobs", "Ipred")]
        .to_frame()
        .reset_index()
    )

    result["file"] = mtzpath.split("/")[0]
    result["spacegroup"] = mtz.spacegroup.xhm()

    return result, labels


def run_analysis(args, show=True):
    results = []
    labels = None

    overall = False

    for m in args.mtzs:
        result,labels = compute_ccpred(m, overall=overall, bins=args.bins, method=args.method)
        if isinstance(result, tuple):
            results.append(result)
            labels = result
        else:
            results.append(result)

    results = rs.concat(results, check_isomorphous=False)
    results = results.reset_index(drop=True)
    results["CCpred"] = results[("Iobs", "Ipred")]
    results.drop(columns=[("Iobs", "Ipred")], inplace=True)

    results['bin'] = results['bin'].to_numpy('int32')
    results['test'] = np.array(['Train', 'Test'])[results['test']]

    if args.output is not None:
        results.to_csv(args.output)

    sns.lineplot(
        data = results,
        x = 'bin',
        y = 'CCpred',
        style = 'test',
        color="k",
    )
    plt.xticks(range(args.bins), labels, rotation=45, ha="right", rotation_mode="anchor")
    plt.ylabel(r"$\mathrm{CC_{pred}}$ " + f"({args.method})")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.xlabel("Resolution ($\mathrm{\AA}$)")
    plt.grid(which='both', axis='both', ls='dashdot')
    plt.tight_layout()
    if show:
        print(results.to_string())
        plt.show()

def main():
    parser = ArgumentParser().parse_args()
    run_analysis(parser, True)

