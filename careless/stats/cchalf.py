"""
Compute CChalf from careless output.
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

        self.add_argument(
            "-m",
            "--method",
            default="pearson",
            choices=["pearson", "spearman", "weighted"],
            help="Method for computing correlation coefficient (pearson, spearman, or weighted). "
            "The 'weighted' option uses a Pearson CC with maximum-likelihood weights. Pearson is the default.",
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

        self.add_argument(
            "--use-structure-factors",
            action="store_true",
            help="Use structure factors instead of intensities for the CChalf calculation.",
        )


def weighted_pearson_ccfunc(df, key="I"):
    k1,k2 = f"{key}1",f"{key}2"
    q1,q2 = f"Sig{key}1",f"Sig{key}2"
    x = df[k1].to_numpy('float32')
    y = df[k2].to_numpy('float32')
    w = np.reciprocal(
        np.square(df[q1]) + np.square(df[q2])
    ).to_numpy('float32')
    return rs.utils.weighted_pearsonr(x, y, w)

def spearman_ccfunc(df, key='I'):
    k1,k2 = f"{key}1",f"{key}2"
    return df[[k1,k2]].corr(method='spearman')[k1][k2]

def pearson_ccfunc(df, key='I'):
    k1,k2 = f"{key}1",f"{key}2"
    return df[[k1,k2]].corr(method='pearson')[k1][k2]

def make_halves_cchalf(mtz, bins=10):
    """Construct half-datasets for computing CChalf"""

    half1 = mtz.loc[mtz.half == 0].copy()
    half2 = mtz.loc[mtz.half == 1].copy()

    # Support anomalous
    if "F(+)" in half1.columns:
        half1 = half1.stack_anomalous()
        half2 = half2.stack_anomalous()

    out = half1[["F", "SigF", "I", "SigI", "repeat"]].merge(
        half2[["F", "SigF", "I", "SigI", "repeat"]], on=["H", "K", "L", "repeat"], suffixes=("1", "2")
    ).dropna()
    return out

def run_analysis(args):
    ds = []
    for m in args.mtz:
        _ds = rs.read_mtz(m)
        _ds = _ds.rename(columns={
            'SIGI' : 'SigI',
            'SIGF' : 'SigF',
        })
        #non-isomorphism could lead to different resolution for each mtz
        #need to calculate dHKL before concatenating 
        _ds = make_halves_cchalf(_ds)
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

    if args.use_structure_factors:
        ds = ds[["file", "bin", "repeat", "F1", "SigF1", "F2", "SigF2", "Spacegroup"]].rename(
            columns={
                'F1' : 'I1',
                'F2' : 'I2',
                'SigF1' : 'SigI1',
                'SigF2' : 'SigI2',
            }
        )

    if args.overall:
        grouper = ds.groupby(["bin", "repeat"])
    else:
        grouper = ds.groupby(["file", "bin", "repeat"])


    if args.method.lower() == "spearman":
        ccfunc = spearman_ccfunc
    elif args.method.lower() == "pearson":
        ccfunc = pearson_ccfunc
    elif args.method.lower() == "weighted":
        ccfunc = weighted_pearson_ccfunc
    else:
        raise ValueError(f"Unrecognized CC --method, {args.method}")

    result = grouper.apply(ccfunc)
    result = rs.DataSet({"CChalf" : result}).reset_index()
    result['Resolution Range (Å)'] = np.array(labels)[result.bin]
    result['Spacegroup'] = grouper['Spacegroup'].apply('first').to_numpy()
    if not args.overall:
        result['file'] = grouper['file'].apply('first').to_numpy()
        result = result[['file', 'repeat', 'Resolution Range (Å)', 'bin', 'Spacegroup', 'CChalf']]
    else:
        result = result[['repeat', 'Resolution Range (Å)', 'bin', 'Spacegroup', 'CChalf']]


    if args.output is not None:
        result.to_csv(args.output)
    else:
        print(result.to_string())
    
    plot_kwargs = {
        'data' : result,
        'x' : 'bin',
        'y' : 'CChalf',
    }

    if args.overall:
        plot_kwargs['color'] = 'k'
    else:
        plot_kwargs['hue'] = 'file'
        plot_kwargs['palette'] = "Dark2"

    plt.figure(figsize=(args.width, args.height))
    sns.lineplot(**plot_kwargs)
    plt.xticks(range(args.bins), labels, rotation=45, ha="right", rotation_mode="anchor")
    plt.ylabel(r"$\mathrm{CC_{1/2}}$ " + f"({args.method})")
    plt.xlabel("Resolution ($\mathrm{\AA}$)")
    plt.grid(which='both', axis='both', ls='dashdot')
    if args.ylim is not None:
        plt.ylim(args.ylim)
    plt.tight_layout()

    if args.image is not None:
        plt.savefig(args.image)

    if args.show:
        plt.show()


def main():
    parser = ArgumentParser().parse_args()
    run_analysis(parser)

