import pathlib
from argparse import ArgumentParser
import reciprocalspaceship as rs

def get_split_friedel_parser():
    parser = ArgumentParser(description="Split an mtz into anomalous half datasets")
    parser.add_argument("unmerged_mtz", type=pathlib.Path)
    parser.add_argument("-p", "--friedel-plus-mtz", help="Output mtz with Plus Friedel mates and centrics. Default 'friedel_plus.mtz'", default="friedel_plus.mtz", type=pathlib.Path)
    parser.add_argument("-m", "--friedel-minus-mtz", help="Output mtz with Minus Friedel mates. Default 'friedel_minus.mtz'", default="friedel_minus.mtz", type=pathlib.Path)
    return parser

def split_friedel(args=None):
    if args is None:
        parser = get_split_friedel_parser()
        args = parser.parse_args()

    ds = rs.read_mtz(str(args.unmerged_mtz))
    if ds.merged:
        raise ValueError(f"Expected an unmerged Mtz, but {args.unmerged_mtz} is merged.")

    # M/ISYM is part of the MTZ specification and can be used
    # to determine the sign of a Friedel mate. More info at:
    # https://www.ccp4.ac.uk/html/mtzformat.html#column-labels-and-standard-names
    # hkl_to_asu adds "M/ISYM" column and preserves row order
    plus = (ds.hkl_to_asu()["M/ISYM"].to_numpy() % 2 == 1)

    #The double-Wilson prior expects all the centrics in the friedel plus file
    centrics = ds.label_centrics().CENTRIC.to_numpy()
    plus_or_centric = plus | centrics

    #from IPython import embed;embed(colors='linux')
    ds[plus_or_centric].write_mtz(str(args.friedel_plus_mtz))
    ds[~plus_or_centric].write_mtz(str(args.friedel_minus_mtz))


def get_combine_friedel_parser():
    parser = ArgumentParser(description="Combine anomalous half datasets merged with careless into a single mtz")
    parser.add_argument("plus_mtz")
    parser.add_argument("minus_mtz")
    parser.add_argument("out_mtz")
    return parser

def combine_friedel(args=None):
    if args is None:
        parser = get_combine_friedel_parser()
        args = parser.parse_args()

    plus = rs.read_mtz(str(args.plus_mtz))
    minus = rs.read_mtz(str(args.minus_mtz))

    # check whether this is a crossvalidation _xval.mtz format file
    # from careless. if so we need to make sure we retain the 'half'
    # and 'repeat' columns to make it compatible with the stats 
    # submodule
    is_xval_mtz = False
    if ('repeat' in plus) and ('half' in plus):
        is_xval_mtz = True

    anom_keys = [
        'F(+)', 'SigF(+)', 'F(-)', 'SigF(-)', 
        'I(+)', 'SigI(+)', 'I(-)', 'SigI(-)', 
        'N(+)', 'N(-)',
    ]

    out = rs.concat([
        plus,
        minus.apply_symop("-x,-y,-z"),
    ])
    def unstack_anomalous(ds):
        """ Unstack and reorder as phenix would expect"""
        return ds.unstack_anomalous()[anom_keys]

    if is_xval_mtz:
        group_keys = ['half', 'repeat']
        cell,sg = out.cell,out.spacegroup
        out = out.groupby(group_keys).apply(unstack_anomalous)
        out.cell,out.spacegroup = cell,sg
    else:
        out = unstack_anomalous(out)

    out.write_mtz(str(args.out_mtz))

