"""
Estimate the Wilson b-factor from unmerged data.

"""
import argparse
import numpy as np
import reciprocalspaceship as rs
import gemmi

import matplotlib.pyplot as plt
import seaborn as sns
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

def estimate_dmin(ds, ikey, sigkey, isigi_cutoff, bins=20):
    if 'dHKL' not in ds:
        ds.compute_dHKL(inplace=True)
    ds,_ = ds.assign_resolution_bins(bins)
    ds['SNR'] = ds[ikey] / ds[sigkey]
    resolution = ds[['bin', 'dHKL']].groupby('bin').mean()
    isigi = ds[['bin', 'SNR']].groupby('bin').mean()
    dmin = resolution[(isigi >= isigi_cutoff).to_numpy()].min().to_numpy()[0]
    return dmin

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
    ds = ds[ds.dHKL <= parser.dmax]

    if parser.dmin is not None:
        dmin = parser.dmin
    else:
        dmin = estimate_dmin(ds, ikey, sigkey, parser.isigi_cutoff, bins=parser.bins)

    ds = ds[ds.dHKL >= dmin]
    intensity = ds[ikey].to_numpy('float32')
    sigma = ds[sigkey].to_numpy('float32')
    dHKL = ds.dHKL.to_numpy('float32')

    ds,labels = ds.assign_resolution_bins(parser.bins)
    xkey = r'$1 / d_{HKL}^2\ (Å^{-2})$'
    ds[xkey] = np.reciprocal(ds.dHKL * ds.dHKL).to_numpy('float32')
    mean = ds.groupby("bin").mean()
    ykey = r'$\log \langle I \rangle$'
    mean[ykey] = np.log(mean[ikey]).to_numpy('float32')

    slope, intercept, r_value, p_value, std_err = linregress(mean[xkey], mean[ykey])

    title = f"Estimated Wilson b-factor: {-2. * slope:0.2f} ± {2. * std_err:0.2f}"
    if parser.plot:
        sns.regplot(data=mean, x=xkey, y=ykey, color='k')
        plt.title(title)
        plt.show()
    print(title)


def main():
    parser = ArgumentParser().parse_args()
    run_analysis(parser)


if __name__=='__main__':
    main()

