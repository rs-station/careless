"""
Compute CChalf from careless output.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import reciprocalspaceship as rs
import seaborn as sns
from scipy.optimize import minimize


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

        self.add_argument(
            "-b",
            "--bins",
            default=10,
            type=int,
            help=("Number of resolution bins to use, the default is 10."),
        )

        self.add_argument(
            "--overall",
            action="store_true",
            help="Pool all prediction mtz files into a single calculation rather than treating each file individually.",
        )

def rsplit(dataset):
    x,y = dataset.F1.to_numpy(),dataset.F2.to_numpy()
    def rfunc(k):
        return np.sum(np.abs(x - k * y)) / np.sum(x + k * y)

    p = minimize(rfunc, 1.)
    return np.sqrt(2) * p.fun 

def make_halves_rsplit(mtz, bins=10):
    """Construct half-datasets for computing Rsplit"""

    half1 = mtz.loc[mtz.half == 0].copy()
    half2 = mtz.loc[mtz.half == 1].copy()

    # Support anomalous
    if "F(+)" in half1.columns:
        half1 = half1.stack_anomalous()
        half2 = half2.stack_anomalous()

    out = half1[["F", "SigF", "repeat"]].merge(
        half2[["F", "SigF", "repeat"]], on=["H", "K", "L", "repeat"], suffixes=("1", "2")
    ).dropna()
    return out

def run_analysis(args):
    ds = []
    for m in args.mtz:
        _ds = rs.read_mtz(m)
        #non-isomorphism could lead to different resolution for each mtz
        #need to calculate dHKL before concatenating 
        _ds = make_halves_rsplit(_ds)
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
    result = grouper.apply(rsplit)
    result = rs.DataSet({"Rsplit" : result}).reset_index()
    result['Resolution Range (Å)'] = np.array(labels)[result.bin]
    result['Spacegroup'] = grouper['Spacegroup'].apply('first').to_numpy()
    if not args.overall:
        result['file'] = grouper['file'].apply('first').to_numpy()
        result = result[['file', 'repeat', 'Resolution Range (Å)', 'bin', 'Spacegroup', 'Rsplit']]
    else:
        result = result[['repeat', 'Resolution Range (Å)', 'bin', 'Spacegroup', 'Rsplit']]


    if args.output is not None:
        result.to_csv(args.output)
    else:
        print(result.to_string())
    
    plot_kwargs = {
        'data' : result,
        'x' : 'bin',
        'y' : 'Rsplit',
    }

    if args.overall:
        plot_kwargs['color'] = 'k'
    else:
        plot_kwargs['hue'] = 'file'
        plot_kwargs['palette'] = "Dark2"

    sns.lineplot(**plot_kwargs)
    plt.xticks(range(args.bins), labels, rotation=45, ha="right", rotation_mode="anchor")
    plt.ylabel(r"$R_{\mathrm{split}}$")
    plt.xlabel("Resolution ($\mathrm{\AA}$)")
    plt.grid(which='both', axis='both', ls='dashdot')
    plt.tight_layout()
    if args.ylim is not None:
        plt.ylim(args.ylim)

    if args.image is not None:
        plt.savefig(args.image)

    if args.show:
        plt.show()


def main():
    parser = ArgumentParser().parse_args()
    run_analysis(parser)

