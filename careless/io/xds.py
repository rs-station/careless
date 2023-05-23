"""
Convert XDS hkl files to mtzs for careless.
"""

import argparse
import numpy as np
import reciprocalspaceship as rs
import gemmi


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(
            formatter_class=argparse.RawTextHelpFormatter, 
            description=__doc__
        )

        self.add_argument(
            "hkl", 
            help="Unmerged HKL file from XDS.",
        )

        self.add_argument(
            "mtz_out", 
            help="Output mtz file name.",
        )

        self.add_argument(
            "-t", 
            "--file-type", 
            default=None,
            type=str,
            help="Override the type of HKL file. This argument can be one of 'ascii' or 'integrate'. "
                 "The default is None which will try to use the header to infer the type.",
        )

        self.add_argument(
            "-s", 
            "--spacegroup", 
            default=None,
            type=str,
            help="Override the space group. This can be a space group number like 19 or a string like 'P 21 21 21'. ",
        )

        self.add_argument(
            "-c", 
            "--cell", 
            default=None,
            nargs=6,
            metavar=("a", "b", "c", "alpha", "beta", "gamma"),
            type=float,
            help="Override the unit cell. Supply six numbers (e.g. --cell 10 20 30 90 90 90).",
        )


def get_unit_cell(file_name):
    for line in open(file_name):
        if line.startswith("!UNIT_CELL_CONSTANTS="):
            cell = [float(i) for i in line.split()[1:]]
            return gemmi.UnitCell(*cell)

def get_space_group(file_name):
    for line in open(file_name):
        if line.startswith("!SPACE_GROUP_NUMBER="):
            sg = line.split()[1]
            return gemmi.SpaceGroup(sg)

def _read_hkl(file_name, cell=None, spacegroup=None, names=None):

    if cell is None:
        cell = get_unit_cell(file_name)
    if spacegroup is None:
        spacegroup = get_space_group(file_name)

    ds = rs.read_csv(
        file_name,
        delim_whitespace=True,
        comment='!',
        names=names,
        cell=cell,
        spacegroup=spacegroup,
        merged=False,
    )
    return ds

def read_integrate_hkl(file_name, cell=None, spacegroup=None):
    cols = [
        "H", "K", "L",
        "IOBS","SIGMA",
        "XCAL", "YCAL", "ZCAL",
        "RLP", "PEAK", "CORR", "MAXC", 
        "XOBS", "YOBS", "ZOBS", 
        "ALF0","BET0","ALF1","BET1","PSI","ISEG"
    ]
    ds = _read_hkl(file_name, cell, spacegroup, cols)
    ds['BATCH'] = ds.ZOBS.round().astype("B")
    return ds

def get_column_names(file_name):
    cols = None
    for line in open(file_name):
        if line.startswith("!NUMBER_OF_ITEMS_IN_EACH_DATA_RECORD="):
            num_cols = int(line.split('=')[1])
            cols = [str(i) for i in range(num_cols)]
        if line.startswith("!ITEM"):
            name = line.split('_')[1].split("=")[0]
            index = int(line.split("=")[1])
            cols[index - 1] = name
        if line.startswith("!END_OF_HEADER"):
            break
    return cols

def read_ascii_hkl(file_name, cell=None, spacegroup=None, zkey='ZD'):
    cols = get_column_names(file_name)
    ds = _read_hkl(file_name, cell, spacegroup, cols)
    if zkey in ds:
        ds['BATCH'] = ds[zkey].round().astype("B")
    return ds

def get_format_field(file_name):
    for line in open(file_name):
        if line.startswith("!FORMAT="):
            return line.split()[0].split('=')[1]
        if line.startswith('!END_OF_HEADER'):
            break
    return None

def infer_file_type(file_name):
    format_name = get_format_field(file_name)
    if format_name == 'XDS_ASCII':
        return 'ascii'
    elif format_name is None:
        return 'integrate'
    else:
        raise ValueError(f"Could not determine filetype for file_name: {file_name}")

def read_hkl(file_name, cell=None, spacegroup=None, file_type=None):
    if file_type is None:
        file_type = infer_file_type(file_name)

    if file_type == 'integrate':
        return read_integrate_hkl(file_name, cell, spacegroup)
    elif file_type == 'ascii':
        return read_ascii_hkl(file_name, cell, spacegroup)
    else:
        raise ValueError(f"file_type, {file_type} not one of 'integrate', 'ascii'.")

def run(parser):
    ds = read_hkl(
        parser.hkl, 
        parser.cell, 
        parser.spacegroup, 
        parser.file_type
    )
    ds.set_index(['H', 'K', 'L']).write_mtz(parser.mtz_out)

def main():
    parser = ArgumentParser().parse_args()
    run(parser)
