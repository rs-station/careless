import numpy as np
import reciprocalspaceship as rs

def expand_harmonics(ds, dmin=None,  wavelength_key='Wavelength', anomalous=False):
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

    #Here's where we get the metadata for Laue harmonic deconvolution
    #This is the HKL of the closest refl on each central ray
    Hobs = ds.loc[:,['Hobs', 'Kobs', 'Lobs']].to_numpy()
    H_0 = (Hobs/np.gcd.reduce(Hobs, axis=1)[:,None]).astype(np.int32)
    ds['H_0'],ds['K_0'],ds['L_0'] = H_0.T

    d_0 = rs.utils.compute_dHKL(H_0, ds.cell)
    ds['d_0'] = rs.DataSeries(d_0, dtype="MTZReal", index=ds.index)
    if dmin is None:
        dmin = ds['dHKL'].min() - 1e-12
    ds['n_max'] =  np.floor(ds['d_0']/dmin).astype(int)

    #This is the harmonic number as indexed
    n_obs = (np.linalg.norm(ds[['Hobs', 'Kobs', 'Lobs']].astype(float), 2, 1) /
        np.linalg.norm(ds[['H_0', 'K_0', 'L_0']].astype(float), 2, 1)).astype(int)

    #Change peak wavelength to correspond to H_0
    # \lambda_n = (1/n) * \lambda_1
    # there is some abuse of notation here \lambda_1 corresponds to H_0
    ds[wavelength_key] = ds[wavelength_key] * n_obs
    ds.loc[:,'Hobs'],ds.loc[:,'Kobs'],ds.loc[:,'Lobs'] = ds['H_0'],ds['K_0'],ds['L_0']

    #This is difficult to read but basically this is where we make the indices to expand
    #each harmonic the appropriate number of times given dmin
    idx = np.repeat(np.arange(len(ds)), ds.n_max)
    n = np.tile(np.arange(1, ds.n_max.max()+2), (len(ds), 1)) 
    n = n[n <= ds.n_max.to_numpy()[:,None]]

    #Expand the harmonics and adjust the wavelength and miller indices to match
    ds = ds.iloc[idx]
    ds['harmonic'] = n
    ds[wavelength_key] = ds[wavelength_key] / n
    ds['Hobs'],ds['Kobs'],ds['Lobs'] = n*ds['H_0'],n*ds['K_0'],n*ds['L_0']
    ds.compute_dHKL(inplace=True)

    #Update the HKLs to reflect the new harmonics
    ds['H'],ds['K'],ds['L'] = ds['Hobs'],ds['Kobs'],ds['Lobs'] 
    ds = ds.set_index(['H','K','L']).hkl_to_asu().reset_index()

    if anomalous:
        friedel_sign = 2 * (ds['M/ISYM'] %2 - 0.5).to_numpy()
        ds.loc[:,['H', 'K', 'L']] = friedel_sign[:,None] * ds.loc[:,['H', 'K', 'L']]

    return ds

