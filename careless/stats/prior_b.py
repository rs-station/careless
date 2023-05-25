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


        self.add_argument(
            "-d",
            "--dmin",
            type=float,
            default=0.,
            help="Minimum resolution cutoff in (Å) which defaults to 0.",
        )

        self.add_argument(
            "-x",
            "--dmax",
            type=float,
            default=np.inf,
            help="Maximum resolution cutoff in (Å) which defaults to infinity.",
        )

        self.add_argument(
            "-c",
            "--isigi-cutoff",
            type=float,
            default=0.,
            help="Minimum I over SigI cutoff which defaults to 0.",
        )


        self.add_argument(
            "--plot",
            action="store_true",
            help="Make a plot of the results and display it using matplotlib.",
        )


def estimate_b_factor(intensity, sigma, dHKL, bins=20):
    """
    log(<I>) ~ -2B * dHKL**-2.

    We can estimate the B factor by taking the slope of the 
    regression of log(<I>) vs 1/d^2. We do this by dividing
    the data into resolution bins and taking the average 
    intensity in each bin. 
    """
    import pandas as pd
    df = pd.DataFrame({
        'inv_d2' : np.reciprocal(dHKL * dHKL),
        'I' : intensity,
        'SigI' : sigma,
    })

    bin_id,edges = rs.utils.bin_by_percentile(dHKL, bins)
    bin_center = edges[np.tile(np.arange(bins), 2)[1:-1].reshape((2,bins-1))].mean(0)

    x = np.reciprocal(np.square(dHKL))

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
    ds = ds[(ds.dHKL >= parser.dmin) & (ds.dHKL <= parser.dmax) & (ds[ikey] / ds[sigkey] >= parser.isigi_cutoff)]
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

    if parser.plot:
        sns.regplot(data=mean, x=xkey, y=ykey, color='k')
        plt.title(f"y= {slope:0.1f}x + {intercept:0.1f}")
        plt.show()


    from IPython import embed
    embed(colors='linux')
    XX
    b = estimate_b_factor(intensity, sigma, dHKL, parser.bins)

def main():
    parser = ArgumentParser().parse_args()
    run_analysis(parser)


if __name__=='__main__':
    main()

