"""
Compute CCanom from careless output.
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

        # Optional arguments
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
            help=("Optionally save CCanom values to this file in csv format."),
        )



def make_halves_ccanom(mtz, bins=10):
    """Construct half-datasets for computing CCanom"""

    half1 = mtz.loc[mtz.half == 0].copy()
    half2 = mtz.loc[mtz.half == 1].copy()

    half1["DF"] = half1["F(+)"] - half1["F(-)"]
    half2["DF"] = half2["F(+)"] - half2["F(-)"]

    temp = half1[["DF", "repeat"]].merge(
        half2[["DF", "repeat"]], on=["H", "K", "L", "repeat"], suffixes=("1", "2")
    )
    temp, labels = temp.assign_resolution_bins(bins)

    return temp, labels


def analyze_ccanom_mtz(mtzpath, bins=10, return_labels=True, method="spearman"):
    """Compute CCanom from 2-fold cross-validation"""

    mtz = rs.read_mtz(mtzpath)

    # Error handling -- make sure MTZ file is appropriate
    if "half" not in mtz.columns:
        raise ValueError("Please provide MTZs from careless crossvalidation")

    if "F(+)" not in mtz.columns:
        raise ValueError("Please provide MTZs merged with `--anomalous` in careless")

    mtz = mtz.acentrics
    mtz = mtz.loc[(mtz["N(+)"] > 0) & (mtz["N(-)"] > 0)]
    m, labels = make_halves_ccanom(mtz, bins=bins)

    grouper = m.groupby(["bin", "repeat"])[["DF1", "DF2"]]
    result = (
        grouper.corr(method=method).unstack()[("DF1", "DF2")].to_frame().reset_index()
    )

    if return_labels:
        return result, labels
    else:
        return result


def run_analysis(args, show=True):
    results = []
    labels = None
    for m in args.mtz:
        result = analyze_ccanom_mtz(m, method=args.method, bins=args.bins)
        if result is None:
            continue
        else:
            result[0]["filename"] = m
            results.append(result[0])
            labels = result[1]

    results = rs.concat(results, check_isomorphous=False)
    results = results.reset_index(drop=True)
    results["CCanom"] = results[("DF1", "DF2")]
    results.drop(columns=[("DF1", "DF2")], inplace=True)

    for k in ('bin', 'repeat'):
        results[k] = results[k].to_numpy('int32')

    if args.output is not None:
        results.to_csv(args.output)

    sns.lineplot(
        data=results, x="bin", y="CCanom", hue="filename", palette="Dark2"
    )
    plt.xticks(range(args.bins), labels, rotation=45, ha="right", rotation_mode="anchor")
    plt.ylabel(r"$\mathrm{CC_{anom}}$ " + f"({args.method})")
    plt.xlabel("Resolution ($\mathrm{\AA}$)")
    plt.grid(which='both', axis='both', ls='dashdot')
    plt.tight_layout()
    if show:
        print(results.to_string())
        plt.show()

def main():
    parser = ArgumentParser().parse_args()
    run_analysis(parser, True)

