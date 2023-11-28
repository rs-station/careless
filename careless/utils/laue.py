import numpy as np
import reciprocalspaceship as rs


def calculate_harmonic(H):
    n = np.gcd.reduce(H, axis=-1)
    return n

@rs.decorators.range_indexed
def expand_harmonics(ds, dmin=None,  wavelength_key='Wavelength'):
    """
    Expand reflection observations to include all contributing harmonics. All 
    contributing reflections will be included out to a resolution cutoff 
    irrespective of peak wavelength.

    Parameters
    ----------
    ds : rs.DataSet
        Laue data without multiples. These should be unmerged data with 
        miller indices that have not been mapped to the ASU (P1 Miller Indices).
    dmin : float
        Highest resolution in Å to which harmonics will be predicted. If not 
        supplied, the highest resolution reflection in ds will set dmin.

    Returns
    -------
    ds : rs.DataSet
        DataSet with all reflection observations expanded to include their 
        constituent reflections. New columns 'H_0', 'K_0', 'L_0' will be added 
        to each reflection to store the Miller indices of the innermost 
        reflection on each central ray. 
    """
    if ds.merged:
        raise ValueError("Expected unmerged data, but ds.merged is True")

    ds = ds.copy()

    #Here's where we get the metadata for Laue harmonic deconvolution
    #This is the HKL of the closest refl on each central ray
    if 'dHKL' not in ds:
        ds.compute_dHKL(inplace=True)
    if dmin is None:
        dmin = ds['dHKL'].min() - 1e-12

    Hobs = ds.get_hkls()

    #Calculated the harmonic as indexed
    nobs = calculate_harmonic(Hobs)

    #Add primary harmonic miller index, wavelength, and resolution
    # H = H_n / n
    # lambda = lambda_n * n
    # d = d_n * n
    H_0 = (Hobs/nobs[:,None]).astype(np.int32)
    d_0 = ds['dHKL'].to_numpy() * nobs
    Wavelength_0 = ds[wavelength_key].to_numpy() * nobs

    #This is the largest harmonic that should be
    #included for each observation in order to
    #respect the resolution cutoff
    n_max =  np.floor_divide(d_0, dmin).astype(int)

    #This is where we make the indices to expand
    #each harmonic the appropriate number of times given dmin
    n = np.arange(n_max.max()) + 1
    idx,n = np.where(n[None,:] <= n_max[:,None])
    n = n + 1
    #idx are the indices for expansion and n is the corresponding
    #set of harmonic integers

    ds = ds.iloc[idx]
    ds.reset_index(inplace=True, drop=True)
    ds['H_0'],ds['K_0'],ds['L_0'] = H_0[idx].T
    ds[wavelength_key] = (Wavelength_0[idx] / n)
    ds['H'],ds['K'],ds['L'] = (n[:,None] * H_0[idx]).T

    #Update dHKL using the cell and not harmonics
    #d_0[idx] / n is not numerically precise enough
    ds.compute_dHKL(inplace=True)

    return ds
