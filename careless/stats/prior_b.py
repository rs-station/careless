"""
Estimate the Wilson b-factor from unmerged data.

"""
import argparse
import numpy as np
import reciprocalspaceship as rs
import pandas as  pd
from scipy.stats import linregress


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(
            formatter_class=argparse.RawTextHelpFormatter, 
            description=__doc__
        )

        # Required arguments
        self.add_argument(
            "input",
            nargs="+",
            help="MTZs or stream files containing unmerged data",
        )

        # Optional arguments
        self.add_argument(
            "-i",
            "--intensity-key",
            type=str,
            default=None,
            help="Intensity column to use. The first one will be used by default.",
        )

        self.add_argument(
            "-s",
            "--sigma-key",
            type=str,
            default=None,
            help="Sigma intensity column to use. The first one will be used by default.",
        )

        self.add_argument(
            "-b",
            "--bins",
            type=int,
            default=20,
            help="Number of bins into which to divide the data.",
        )

        resolution_group = self.add_mutually_exclusive_group()

        resolution_group.add_argument(
            "-c",
            "--isigi-cutoff",
            type=float,
            default=1.5,
            help="If this option is supplied, the script tries to estimate an appropriate resolution" 
                 "cutoff from the signal to noise in resolution bins. The default value is 1.5 (I/SigI).",
        )

        resolution_group.add_argument(
            "-d",
            "--dmin",
            type=float,
            default=None,
            help="Minimum resolution cutoff in (Å) which overrides the more automated --isigi-cutoff.",
        )

        self.add_argument(
            "-x",
            "--dmax",
            type=float,
            default=np.inf,
            help="Maximum resolution cutoff in (Å) which defaults to infinity.",
        )

        self.add_argument(
            "--plot",
            action="store_true",
            help="Make a plot of the results and display it using matplotlib.",
        )

def _make_df(dHKL, I, SigI, bins=None):
    df = pd.DataFrame({
        'dHKL' : dHKL,
        'I' : I,
        'SigI' : SigI,
    })
    if bins is not None:
        labels, edges = rs.utils.bin_by_percentile(dHKL, bins)
        df['bin'] = labels
    return df

def _truncate_data(dHKL, I, SigI, bins=20, isigi_cutoff=None, dmin=None):
    if isigi_cutoff is None and dmin is None:
        return dHKL, I, SigI
    if dmin is not None:
        idx = dHKL >= dmin
        return dHKL[idx], I[idx], SigI[idx]
    df = _make_df(dHKL, I, SigI, bins)
    df['isigi'] = df.I / df.SigI
    mu = df.groupby('bin').mean()
    mu.isigi > isigi_cutoff
    dmin = mu.dHKL[mu.isigi >= isigi_cutoff].min()
    return _truncate_data(dHKL, I, SigI, bins, None, dmin)

def estimate_b(dHKL, I, SigI, bins=20, isigi_cutoff=None, dmin=None):
    """
    Estimate the Wilson b-factor for the data. 

    Parameters
    ----------
    dHKL : array
        The resolution of each reflection in Å.
    I : array
        The intensity of each reflection.
    SigI : array
        The uncertainty of each reflection.
    bins : int (optional)
        The number of bins to use for fitting the wilson plot
    isigi_cutoff : float (optional)
        Optionally truncate the resolution by I / SigI
    dmin : float (optional)
        Optionally truncate the resolution at a specific value in Å.
        This option supersedes isigi_cutoff
    """
    dHKL, I, SigI = _truncate_data(dHKL, I, SigI, bins, isigi_cutoff, dmin)
    df = _make_df(dHKL, I, SigI, bins)
    df['VarI'] = np.square(df.SigI)
    df['inv_d2'] = np.reciprocal(np.square(df['dHKL']))
    mu = df[['bin', 'I', 'inv_d2']].groupby('bin').mean()
    x,y = mu.inv_d2, np.log(mu.I)

    keys = ('slope', 'intercept', 'r_value', 'p_value', 'stderr', 'intercept_stderr', 'x', 'y')
    result = linregress(x, y) 
    result.x = x
    result.y = y
    return result

def run_analysis(parser):
    from careless.io.formatter import get_first_key_of_dtype
    ds = []
    for i,file in enumerate(parser.input):
        if file.endswith('.mtz'):
            _ds = rs.read_mtz(file)
        elif file.endswith('.stream'):
            _ds = rs.read_crystfel(file)

        _ds['file_id'] = i
        ds.append(_ds)
    ds = rs.concat(ds)

    ikey = parser.intensity_key
    sigkey = parser.sigma_key
    if ikey is None:
        ikey = get_first_key_of_dtype(ds, 'J')
    if sigkey is None:
        sigkey = get_first_key_of_dtype(ds, 'Q')
        for key in ds:
            if ds[key].dtype == 'Q' and key.endswith(ikey):
                sigkey = key

    ds.compute_dHKL(inplace=True)

    dHKL,I,SigI = ds[['dHKL', 'I', 'SigI']].to_numpy('float32').T
    fit = estimate_b(dHKL, I, SigI, parser.bins, parser.isigi_cutoff, parser.dmin)


    title = f"Estimated Wilson b-factor: {-2. * fit.slope:0.2f} ± {2. * fit.stderr:0.2f}"
    if parser.plot:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.regplot(x=fit.x, y=fit.y, color='k')
        plt.xlabel(r'$1 / d_{HKL}^2\ (Å^{-2})$')
        plt.ylabel(r'$\log \langle I \rangle$')
        plt.title(title)
        plt.show()
    print(title)


def main():
    parser = ArgumentParser().parse_args()
    run_analysis(parser)


if __name__=='__main__':
    main()

