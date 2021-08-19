import pytest
from os import listdir
from os.path import dirname, abspath, join
import numpy as np
import pandas as pd
import re
import reciprocalspaceship as rs
import gemmi

@pytest.fixture
def cell_and_spacegroups():
    data = [
        ((10., 20., 30., 90., 80., 75.), 'P 1'),
        ((30., 50., 80., 90., 100., 90.), 'P 1 21 1'),    
        ((10., 20., 30., 90., 90., 90.), 'P 21 21 21'),
        ((89., 89., 105., 90., 90., 120.), 'P 31 2 1'),
        ((30., 30., 30., 90., 90., 120.), 'R 32'),
    ]
    return [(gemmi.UnitCell(*i), gemmi.SpaceGroup(j)) for i,j in data]


def load_dataset(datapath):
    """
    Load dataset at given datapath. Datapath is expected to be a list of
    directories to follow.
    """
    inFN = abspath(join(dirname(__file__), datapath))
    return rs.read_mtz(inFN)
    

@pytest.fixture
def laue_inputs():
    from careless.utils.laue import expand_harmonics
    from careless.io.asu import ReciprocalASU,ReciprocalASUCollection

    ds = load_dataset('data/pyp_off.mtz')
    ds.loc[:,['Hobs', 'Kobs', 'Lobs']] = ds.get_hkls()
    ds.hkl_to_asu(inplace=True)

    #expand the wavelength range a bit to get more harmonics for testing
    lam_min = 0.8 * ds.Wavelength.min()
    lam_max = 1.2 * ds.Wavelength.max()

    ds = expand_harmonics(ds)
    ds = ds[(ds.Wavelength >= lam_min) & (ds.Wavelength <= lam_max)]
    hkls = ds.get_hkls()

    rasu = ReciprocalASU(ds.cell, ds.spacegroup, ds.compute_dHKL().dHKL.min(), False)
    rasu_collection = ReciprocalASUCollection([rasu])

    refl_id  = rasu_collection.to_refl_id(np.zeros((len(hkls), 1), dtype='int32'), hkls)
    image_id = ds.groupby('BATCH').ngroup().to_numpy('int32')[:,None]
    metadata = ds[[
        'Wavelength',
        'dHKL',
        'Hobs',
        'Kobs',
        'Lobs',
        'BATCH'
    ]].to_numpy('float32')
    intensities   = ds.I.to_numpy('float32')[:,None]
    uncertainties = ds.SigI.to_numpy('float32')[:,None]
    wavelength  = ds.Wavelength.to_numpy('float32')[:,None]
    harmonic_id = ds.groupby(['BATCH', 'H_0', 'K_0', 'L_0']).ngroup().to_numpy('int32')[:,None]

    inputs = [
        refl_id,
        image_id,
        metadata,
        intensities,
        uncertainties,
        wavelength,
        harmonic_id
    ]
    return inputs
