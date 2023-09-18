import numpy as np
import reciprocalspaceship as rs


def expand_harmonics(ds, dmin=None,  wavelength_key='Wavelength'):
    """
    Expand reflection observations to include all contributing harmonics. All 
    contributing reflections will be included out to a resolution cutoff 
    irrespective of peak wavelength.

    Parameters
    ----------
    ds : rs.DataSet
        Laue data without multiples. Must have 'Hobs', 'Kobs', and 'Lobs' columns.
        These must correspond to the observed miller indices not those in the ASU.
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
    if 'Hobs' not in ds:
        raise KeyError("Expected 'Hobs' column in ds, but no 'Hobs' found")
    if 'Kobs' not in ds:
        raise KeyError("Expected 'Kobs' column in ds, but no 'Kobs' found")
    if 'Lobs' not in ds:
        raise KeyError("Expected 'Lobs' column in ds, but no 'Lobs' found")

    ds = ds.copy()
    if 'H' not in ds:
        ds.reset_index(inplace=True)
    if 'H' not in ds:
        raise ValueError("No column 'H' in index or columns")

    #Here's where we get the metadata for Laue harmonic deconvolution
    #This is the HKL of the closest refl on each central ray
    if 'dHKL' not in ds:
        ds.compute_dHKL(inplace=True)

    Hobs = ds.loc[:,['Hobs', 'Kobs', 'Lobs']].to_numpy(np.int32)

    #Calculated the harmonic as indexed
    nobs = np.gcd.reduce(Hobs, axis=1)

    #Add primary harmonic miller index, wavelength, and resolution
    # H = H_n / n
    # lambda = lambda_n * n
    # d = d_n * n
    H_0 = (Hobs/nobs[:,None]).astype(np.int32)
    ds['H_0'],ds['K_0'],ds['L_0'] = H_0.T
    ds['d_0'] = ds['dHKL']*nobs
    ds['Wavelength_0'] = ds[wavelength_key]*nobs
    ds['nobs'] = nobs

    if dmin is None:
        dmin = ds['dHKL'].min() - 1e-12
    ds['n_max'] =  np.floor_divide(ds['d_0'], dmin).astype(int)

    #This is where we make the indices to expand
    #each harmonic the appropriate number of times given dmin
    n = np.arange(1, ds.n_max.max() + 2)
    idx,n = np.where(n <= ds.n_max.to_numpy()[:,None])
    n = n + 1

    #Expand the harmonics and adjust the wavelength and miller indices to match
    ds = ds.iloc[idx]
    ds['harmonic'] = n
    ds['Hobs'],ds['Kobs'],ds['Lobs'] = n*ds['H_0'],n*ds['K_0'],n*ds['L_0']

    #Update the HKLs to reflect the new harmonics
    H = ds.get_hkls()
    H_0_asu = (H/np.gcd.reduce(H, axis=1)[:,None]).astype(np.int32)
    ds.loc[:,['H', 'K', 'L']] = n[:,None] * H_0_asu
    ds['dHKL'] = ds['d_0'] / n
    ds[wavelength_key] = ds['Wavelength_0'] / n
    ds.reset_index(inplace=True, drop=True)

    return ds
