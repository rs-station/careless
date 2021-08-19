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

