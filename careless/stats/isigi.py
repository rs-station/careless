"""
Compute I/sigI from careless output.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import reciprocalspaceship as rs
import seaborn as sns
import os


from careless.io.formatter import get_first_key_of_dtype
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
            default=20,
            type=int,
            help=("Number of resolution bins to use, the default is 20."),
        )

        self.add_argument(
            "--intensity-key",
            dest="I_col",
            default=None,
            type=str,
            help=("Intensity key"),
        )
        self.add_argument(
            "--uncertainty-key",
            dest="sigI_col",
            default=None,
            type=str,
            help=("Sigma(Intensity) key"),
        )
        self.add_argument(
            "--log",
            action="store_true",
            help=("Use a logarithmic scale for the y-axis."),
        )

        self.add_argument(
            "--overall",
            action="store_true",
            help="Pool all prediction mtz files into a single calculation rather than treating each file individually.",
        )


def run_analysis(args):
    ds = []
    for m in args.mtz:
        _ds = rs.read_mtz(m)
        print(m)
        #non-isomorphism could lead to different resolution for each mtz
        #need to calculate dHKL before concatenating 
        _ds.compute_dHKL(inplace=True)
        if len(m)<50:
            _ds['file'] = m
        else:
            _ds['file'] = os.path.basename(m)
        _ds['Spacegroup'] = _ds.spacegroup.xhm()
        ds.append(_ds)
    ds = rs.concat(ds, check_isomorphous=False)
    bins,edges = rs.utils.bin_by_percentile(ds.dHKL, args.bins, ascending=False)
    ds['bin'] = bins
    labels = [
        f"{e1:0.2f} - {e2:0.2f}"
        for e1, e2 in zip(edges[:-1], edges[1:])
    ]

    ikey = args.I_col
    if ikey is None:
        ikey = get_first_key_of_dtype(ds, 'J')
    sigkey=args.sigI_col
    if sigkey is None:
        sigkey = get_first_key_of_dtype(ds, 'Q')
        
    if args.overall:
        grouper = ds.groupby(["bin"])
    else:
        grouper = ds.groupby(["file", "bin"])

    result = grouper.apply(lambda x : np.mean(x[ikey]/x[sigkey]))
    result = rs.DataSet({"I/sigI" : result}).reset_index()
    result['Resolution Range (Å)'] = np.array(labels)[result.bin]
    result['Spacegroup'] = grouper['Spacegroup'].apply('first').to_numpy()
    if not args.overall:
        result['file'] = grouper['file'].apply('first').to_numpy()
        result = result[['file', 'Resolution Range (Å)', 'bin', 'Spacegroup', 'I/sigI']]
    else:
        result = result[['Resolution Range (Å)', 'bin', 'Spacegroup', 'I/sigI']]


    if args.output is not None:
        result.to_csv(args.output)
    else:
        print(result.to_string())
    
    plot_kwargs = {
        'data' : result,
        'x' : 'bin',
        'y' : 'I/sigI',
    }

    if args.overall:
        plot_kwargs['color'] = 'k'
    else:
        plot_kwargs['hue'] = 'file'
        plot_kwargs['palette'] = "Dark2"

    ax=sns.lineplot(**plot_kwargs)
    if args.log:
        ax.set(yscale='log')
    plt.xticks(range(args.bins), labels, rotation=45, ha="right", rotation_mode="anchor")
    plt.ylabel(r"$\mathrm{I/\sigma(I)}$ ")
    plt.xlabel("Resolution ($\mathrm{\AA}$)")
    plt.grid(which='both', axis='both', ls='dashdot')
    plt.ylim(args.ylim)

    plt.tight_layout()

    if args.image is not None:
        plt.savefig(args.image)

    if args.show:
        plt.show()


def main():
    parser = ArgumentParser().parse_args()
    # print(parser)
    run_analysis(parser)

if __name__ == "__main__":
    main()
