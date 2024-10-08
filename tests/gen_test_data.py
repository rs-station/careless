"""
Generate the test data required for running the careless test suite. 
It is required to run this script prior to calling pytest. Here is
an example of how to run the careless tests. 

```
cd careless
pip install -e .[dev]
python tests/gen_test_data.py
pytest
```

"""

from os import listdir,mkdir
from os.path import dirname, abspath, join, exists
import numpy as np
import pandas as pd
import re
import reciprocalspaceship as rs
import gemmi



def main():
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

if __name__=="__main__":
    main()
