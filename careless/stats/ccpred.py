"""
Compute CCpred from careless output.

"""
import argparse
import numpy as np
import reciprocalspaceship as rs
import gemmi

import matplotlib.pyplot as plt
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
            help="MTZ(s) containing prediction data from careless",
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

def weighted_pearson_ccfunc(df, iobs='Iobs', ipred='Ipred', sigiobs='SigIobs'):
    x = df[iobs].to_numpy('float32')
    y = df[ipred].to_numpy('float32')
    w = np.reciprocal(np.square(df[sigiobs])).to_numpy('float32')
    return rs.utils.weighted_pearsonr(x, y, w)

def spearman_ccfunc(df, iobs='Iobs', ipred='Ipred'):
    return df[[iobs, ipred]].corr(method='spearman')[iobs][ipred]

def run_analysis(args):
    labels = None

    overall = False

    ds = []
    for m in args.mtz:
        _ds = rs.read_mtz(m)
        #non-isomorphism could lead to different resolution for each mtz
        #need to calculate dHKL before concatenating 
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
        grouper = ds.groupby(["bin", "test"])
    else:
        grouper = ds.groupby(["file", "bin", "test"])

    if args.method.lower() == "spearman":
        ccfunc = spearman_ccfunc
    elif args.method.lower() == "pearson":
        ccfunc = weighted_pearson_ccfunc

    result = grouper.apply(ccfunc)
    result = rs.DataSet({"CCpred" : result}).reset_index()
    result['Resolution Range (Å)'] = np.array(labels)[result.bin]
    result['Spacegroup'] = grouper['Spacegroup'].apply('first').to_numpy()
    if not args.overall:
        result['file'] = grouper['file'].apply('first').to_numpy()
        result = result[['file', 'Resolution Range (Å)', 'bin', 'test', 'Spacegroup', 'CCpred']]
    else:
        result = result[['Resolution Range (Å)', 'bin', 'test', 'Spacegroup', 'CCpred']]

    result['bin'] = result['bin'].to_numpy('int32')
    result['test'] = np.array(['Train', 'Test'])[result['test']]

    if args.output is not None:
        result.to_csv(args.output)
    else:
        print(result.to_string())

    plot_kwargs = {
        'data' : result,
        'x' : 'bin',
        'y' : 'CCpred',
        'style' : 'test',
    }

    if args.overall:
        plot_kwargs['color'] = 'k'
    else:
        plot_kwargs['hue'] = 'file'
        plot_kwargs['palette'] = "Dark2"

    sns.lineplot(**plot_kwargs)

    plt.xticks(range(args.bins), labels, rotation=45, ha="right", rotation_mode="anchor")
    plt.ylabel(r"$\mathrm{CC_{pred}}$ " + f"({args.method})")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
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

