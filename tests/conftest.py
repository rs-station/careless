import pytest
from os import listdir,mkdir
from os.path import dirname, abspath, join, exists
import numpy as np
import pandas as pd
import re
import reciprocalspaceship as rs
import gemmi

def pytest_sessionstart(session):
    rundir = "data/"
    rundir = abspath(join(dirname(__file__), rundir))

    command = """
    careless poly 
        --disable-progress-bar 
        --iterations=10 
        --merge-half-datasets 
        --half-dataset-repeats=3 
        --test-fraction=0.1 
        --disable-gpu 
        --anomalous 
        --wavelength-key=Wavelength
        dHKL,Hobs,Kobs,Lobs,Wavelength
        pyp_off.mtz 
        pyp_2ms.mtz 
        output/pyp
    """
    if not exists(f"{rundir}/output"):
        mkdir(f"{rundir}/output")
        from subprocess import call
        call(command.split(), cwd=rundir)

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


@pytest.fixture
def xval_mtz():
    datapath = "data/output/pyp_xval_0.mtz"
    filename = abspath(join(dirname(__file__), datapath))
    return filename

@pytest.fixture
def predictions_mtz():
    datapath = "data/output/pyp_predictions_0.mtz"
    filename = abspath(join(dirname(__file__), datapath))
    return filename

@pytest.fixture
def merged_mtz():
    datapath = "data/output/pyp_0.mtz"
    filename = abspath(join(dirname(__file__), datapath))
    return filename

@pytest.fixture
def stream_file():
    datapath = "data/crystfel.stream"
    filename = abspath(join(dirname(__file__), datapath))
    return filename

@pytest.fixture
def off_file():
    datapath = "data/pyp_off.mtz"
    filename = abspath(join(dirname(__file__), datapath))
    return filename


@pytest.fixture
def on_file():
    datapath = "data/pyp_2ms.mtz"
    filename = abspath(join(dirname(__file__), datapath))
    return filename

@pytest.fixture
def on_file_alt_sg():
    datapath = "data/pyp_2ms_P3.mtz"
    filename = abspath(join(dirname(__file__), datapath))
    return filename

def load_dataset(datapath):
    """
    Load dataset at given datapath. Datapath is expected to be a list of
    directories to follow.
    """
    inFN = abspath(join(dirname(__file__), datapath))
    return rs.read_mtz(inFN)

class LaueTestData():
    def __init__(self, mtz_file):
        from careless.utils.laue import expand_harmonics
        from careless.io.asu import ReciprocalASU,ReciprocalASUCollection

        ds = load_dataset(mtz_file)
        ds.loc[:,['Hobs', 'Kobs', 'Lobs']] = ds.get_hkls()
        ds.hkl_to_asu(inplace=True)
        ds.compute_dHKL(inplace=True)

        #expand the wavelength range a bit to get more harmonics for testing
        lam_min = 0.8 * ds.Wavelength.min()
        lam_max = 1.2 * ds.Wavelength.max()

        ds = expand_harmonics(ds)
        ds = ds[(ds.Wavelength >= lam_min) & (ds.Wavelength <= lam_max)]
        hkls = ds.get_hkls()

        rasu = ReciprocalASU(ds.cell, ds.spacegroup, ds.compute_dHKL().dHKL.min(), False)
        rasu_collection = ReciprocalASUCollection([rasu])

        refl_id  = rasu_collection.to_refl_id(np.zeros((len(hkls), 1), dtype='int64'), hkls)[:,None]
        image_id = ds.groupby('BATCH').ngroup().to_numpy('int64')[:,None]
        metadata = ds[[
            'Wavelength',
            'dHKL',
            'Hobs',
            'Kobs',
            'Lobs',
            'BATCH'
        ]].to_numpy('float32')
        intensities = ds.groupby(['BATCH', 'H_0', 'K_0', 'L_0']).first().I.to_numpy('float32')[:,None]
        uncertainties = ds.groupby(['BATCH', 'H_0', 'K_0', 'L_0']).first().SigI.to_numpy('float32')[:,None]
        wavelength  = ds.Wavelength.to_numpy('float32')[:,None]
        harmonic_id = ds.groupby(['BATCH', 'H_0', 'K_0', 'L_0']).ngroup().to_numpy('int64')[:,None]
        pad = len(refl_id) - len(intensities)
        intensities   = np.pad(intensities, [[0,pad],[0,0]])
        uncertainties = np.pad(uncertainties, [[0,pad],[0,0]], constant_values = 1./np.sqrt(np.pi * 2))

        self.data = ds
        self.reciprocal_asu = rasu
        self.reciprocal_asu_collection = rasu_collection
        file_id = np.zeros_like(refl_id)
        self.inputs = [
            refl_id,
            image_id,
            file_id,
            metadata,
            intensities,
            uncertainties,
            wavelength,
            harmonic_id
        ]

laue_test_data = LaueTestData('data/pyp_off.mtz')


@pytest.fixture
def laue_inputs():
    return laue_test_data.inputs

@pytest.fixture
def laue_data_set():
    return laue_test_data.data

@pytest.fixture
def laue_reciprocal_asu_collection():
    return laue_test_data.reciprocal_asu_collection

class MonoTestData():
    def __init__(self, mtz_file):
        #shh these are actually laue data 0.o
        from careless.io.asu import ReciprocalASU,ReciprocalASUCollection

        ds = load_dataset('data/pyp_off.mtz')
        ds.loc[:,['Hobs', 'Kobs', 'Lobs']] = ds.get_hkls()
        ds.hkl_to_asu(inplace=True)
        ds.compute_dHKL(inplace=True)

        hkls = ds.get_hkls()

        rasu = ReciprocalASU(ds.cell, ds.spacegroup, ds.compute_dHKL().dHKL.min(), False)
        rasu_collection = ReciprocalASUCollection([rasu])

        refl_id  = rasu_collection.to_refl_id(np.zeros((len(hkls), 1), dtype='int64'), hkls)[:,None]
        image_id = ds.groupby('BATCH').ngroup().to_numpy('int64')[:,None]
        metadata = ds[[
            'dHKL',
            'Hobs',
            'Kobs',
            'Lobs',
            'BATCH'
        ]].to_numpy('float32')
        intensities   = ds.I.to_numpy('float32')[:,None]
        uncertainties = ds.SigI.to_numpy('float32')[:,None]

        self.data = ds
        self.reciprocal_asu = rasu
        self.reciprocal_asu_collection = rasu_collection
        file_id = np.zeros_like(refl_id)
        self.inputs = [
            refl_id,
            image_id,
            file_id,
            metadata,
            intensities,
            uncertainties,
        ]

mono_test_data = MonoTestData('data/pyp_off.mtz')

@pytest.fixture
def mono_inputs():
    return mono_test_data.inputs

@pytest.fixture
def mono_reciprocal_asu_collection():
    return mono_test_data.reciprocal_asu_collection

@pytest.fixture
def mono_data_set():
    return mono_test_data.data

