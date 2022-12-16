"""
Compute CChalf from careless output.
"""
import argparse
import matplotlib.pyplot as plt
import reciprocalspaceship as rs
import seaborn as sns


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
            "-m",
            "--method",
            default="spearman",
            choices=["spearman", "pearson"],
            help=("Method for computing correlation coefficient (spearman or pearson)"),
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
            help=("Optionally save CChalf values to this file in csv format."),
        )


def make_halves_cchalf(mtz, bins=10):
    """Construct half-datasets for computing CChalf"""

    half1 = mtz.loc[mtz.half == 0].copy()
    half2 = mtz.loc[mtz.half == 1].copy()

    # Support anomalous
    if "F(+)" in half1.columns:
        half1 = half1.stack_anomalous()
        half2 = half2.stack_anomalous()

    temp = half1[["F", "repeat"]].merge(
        half2[["F", "repeat"]], on=["H", "K", "L", "repeat"], suffixes=("1", "2")
    )
    temp, labels = temp.assign_resolution_bins(bins)

    return temp, labels


def analyze_cchalf_mtz(mtzpath, bins=10, return_labels=True, method="spearman"):
    """Compute CChalf from 2-fold cross-validation"""

    mtz = rs.read_mtz(mtzpath)

    # Error handling -- make sure MTZ file is appropriate
    if "half" not in mtz.columns:
        raise ValueError("Please provide MTZs from careless crossvalidation")

    m, labels = make_halves_cchalf(mtz, bins)

    grouper = m.groupby(["bin", "repeat"])[["F1", "F2"]]
    result = (
        grouper.corr(method=method).unstack()[("F1", "F2")].to_frame().reset_index()
    )

    if return_labels:
        return result, labels
    else:
        return result


def run_analysis(args, show=True):
    results = []
    labels = None
    for m in args.mtz:
        result = analyze_cchalf_mtz(m, bins=args.bins, method=args.method)
        if result is None:
            continue
        else:
            result[0]["filename"] = m
            results.append(result[0])
            labels = result[1]

    results = rs.concat(results, check_isomorphous=False)
    results = results.reset_index(drop=True)
    results["CChalf"] = results[("F1", "F2")]
    results.drop(columns=[("F1", "F2")], inplace=True)


    for k in ('bin', 'repeat'):
        results[k] = results[k].to_numpy('int32')

    if args.output is not None:
        results.to_csv(args.output)
    
    sns.lineplot(
        data=results, x="bin", y="CChalf", hue="filename", palette="viridis"
    )
    plt.xticks(range(args.bins), labels, rotation=45, ha="right", rotation_mode="anchor")
    plt.ylabel(r"$\mathrm{CC_{1/2}}$ " + f"({args.method})")
    plt.xlabel("Resolution ($\mathrm{\AA}$)")
    plt.grid(which='both', axis='both', ls='dashdot')
    plt.tight_layout()
    if show:
        print(results.to_string())
        plt.show()


def main():
    parser = ArgumentParser().parse_args()
    run_analysis(parser, True)

