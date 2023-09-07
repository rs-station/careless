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
            "--height",
            default=3,
            help="Height of the plot to make with default value 3 (inches)."
        )

        self.add_argument(
            "--width",
            default=7,
            help="Width of the plot to make with default value 7 (inches)."
        )

def weighted_pearson_ccfunc(df, iobs='Iobs', ipred='Ipred', sigiobs='SigIobs'):
    x = df[iobs].to_numpy('float32')
    y = df[ipred].to_numpy('float32')
    w = np.reciprocal(np.square(df[sigiobs])).to_numpy('float32')
    return rs.utils.weighted_pearsonr(x, y, w)

def spearman_ccfunc(df, iobs='Iobs', ipred='Ipred'):
    return df[[iobs, ipred]].corr(method='spearman')[iobs][ipred]

def run_analysis(args):
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

    ds['BATCH'] = (ds['image_id'] - ds.groupby("asu_id").transform("min")["image_id"] + 1).to_numpy()
    grouper = ds.groupby(["file", "BATCH"])

    if args.method.lower() == "spearman":
        ccfunc = spearman_ccfunc
    elif args.method.lower() == "pearson":
        ccfunc = weighted_pearson_ccfunc

    result = grouper.apply(ccfunc)
    result = rs.DataSet({"CCpred" : result}).reset_index()
    result['file_id'] = grouper.first()['file_id'].to_numpy()
    result['asu_id'] = grouper.first()['asu_id'].to_numpy()
    result = result[['file', 'file_id', 'asu_id', 'BATCH', 'CCpred']]

    if args.output is not None:
        result.to_csv(args.output)
    else:
        print(result.to_string())


    plot_kwargs = {
        'data' : result,
        'x' : 'BATCH',
        'y' : 'CCpred',
        'hue' : 'file',
        'marker' : '.',
        'linestyle' : 'none',
        'palette' : "Dark2",
    }

    plt.figure(figsize=(args.width, args.height))
    sns.lineplot(**plot_kwargs)

    plt.ylabel(r"$\mathrm{CC_{pred}}$ " + f"({args.method})")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid(which='both', axis='both', ls='dashdot')
    plt.tight_layout()

    if args.image is not None:
        plt.savefig(args.image)

    if args.show:
        plt.show()

def main():
    parser = ArgumentParser().parse_args()
    run_analysis(parser)

