"""
Compute CCanom from careless output.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import reciprocalspaceship as rs
import seaborn as sns


from careless.stats.parser import BaseParser
class ArgumentParser(BaseParser):
    def __init__(self):
        super().__init__(
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
            default="pearson",
            choices=["pearson", "spearman"],
            help="Method for computing correlation coefficient (spearman or pearson). "
            "The Pearson CC uses maximum-likelihood weights. Pearson is the default.",
        )

        self.add_argument(
            "-b",
            "--bins",
            default=10,
            type=int,
            help="Number of resolution bins to use, the default is 10.",
        )

        self.add_argument(
            '--overall',
            action="store_true",
            help="Pool all prediction mtz files into a single calculation rather than treating each file individually.",
        )

def make_halves_ccanom(mtz, bins=10):
    """Construct half-datasets for computing CCanom"""
    #Only user rows with both F+ and F-
    mtz = mtz.loc[(mtz["N(+)"] > 0) & (mtz["N(-)"] > 0)]

    half1 = mtz.loc[mtz.half == 0].copy()
    half2 = mtz.loc[mtz.half == 1].copy()

    half1["Danom"] = half1["F(+)"] - half1["F(-)"]
    half2["Danom"] = half2["F(+)"] - half2["F(-)"]
    half1["SigDanom"] = np.sqrt(
        np.square(half1["SigF(+)"]) + np.square(half1["SigF(-)"])
    )
    half2["SigDanom"] = np.sqrt(
        np.square(half2["SigF(+)"]) + np.square(half2["SigF(-)"])
    )

    out = half1[["Danom", "SigDanom", "repeat"]].merge(
        half2[["Danom", "SigDanom", "repeat"]], on=["H", "K", "L", "repeat"], suffixes=("1", "2")
    )
    return out

def weighted_pearson_ccfunc(df):
    x = df['Danom1'].to_numpy('float32')
    y = df['Danom2'].to_numpy('float32')
    w = np.reciprocal(
        np.square(df['SigDanom1']) + np.square(df['SigDanom2'])
    ).to_numpy('float32')
    return rs.utils.weighted_pearsonr(x, y, w)

def spearman_ccfunc(df):
    return df[['Danom1', 'Danom2']].corr(method='spearman')['Danom1']['Danom2']

def run_analysis(args):
    ds = []
    for m in args.mtz:
        _ds = rs.read_mtz(m)
        #non-isomorphism could lead to different resolution for each mtz
        #need to calculate dHKL before concatenating 
        _ds = make_halves_ccanom(_ds)
        _ds.compute_dHKL(inplace=True)
        _ds['file'] = m
        _ds['Spacegroup'] = _ds.spacegroup.xhm()
        ds.append(_ds)
    ds = rs.concat(ds, check_isomorphous=False)
    bins,edges = rs.utils.bin_by_percentile(ds.dHKL, args.bins, ascending=False)
    ds['bin'] = bins
    labels = [
        f"{e1:0.2f} - {e2:0.2f}"
        for e1, e2 in zip(edges[:-1], edges[1:])
    ]

    if args.overall:
        grouper = ds.groupby(["bin", "repeat"])
    else:
        grouper = ds.groupby(["file", "bin", "repeat"])
    if args.method.lower() == "spearman":
        ccfunc = spearman_ccfunc
    elif args.method.lower() == "pearson":
        ccfunc = weighted_pearson_ccfunc
    result = grouper.apply(ccfunc)
    result = rs.DataSet({"CCanom" : result}).reset_index()
    result['Resolution Range (Å)'] = np.array(labels)[result.bin]
    result['Spacegroup'] = grouper['Spacegroup'].apply('first').to_numpy()
    if not args.overall:
        result['file'] = grouper['file'].apply('first').to_numpy()
        result = result[['file', 'repeat', 'Resolution Range (Å)', 'bin', 'Spacegroup', 'CCanom']]
    else:
        result = result[['repeat', 'Resolution Range (Å)', 'bin', 'Spacegroup', 'CCanom']]

    if args.output is not None:
        result.to_csv(args.output)
    else:
        print(result.to_string())

    plot_kwargs = {
        'data' : result,
        'x' : 'bin',
        'y' : 'CCanom',
    }

    if args.overall:
        plot_kwargs['color'] = 'k'
    else:
        plot_kwargs['hue'] = 'file'
        plot_kwargs['palette'] = "Dark2"

    sns.lineplot(**plot_kwargs)
 
    plt.xticks(range(args.bins), labels, rotation=45, ha="right", rotation_mode="anchor")
    plt.ylabel(r"$\mathrm{CC_{anom}}$ " + f"({args.method})")
    plt.xlabel("Resolution ($\mathrm{\AA}$)")
    plt.grid(which='both', axis='both', ls='dashdot')
    plt.tight_layout()

    if args.image is not None:
        plt.savefig(args.image)

    if args.show:
        plt.show()

def main():
    parser = ArgumentParser().parse_args()
    run_analysis(parser)

