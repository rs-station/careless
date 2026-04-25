from argparse import ArgumentParser
import reciprocalspaceship as rs

def get_split_friedel_parser():
    parser = ArgumentParser("Split an mtz into anomalous half-datasets")
    parser.add_argument("unmerged_mtz")
    parser.add_argument("-p", "--friedel-plus-mtz", help="Output mtz with Plus Friedel mates and centrics. Default 'friedel_plus.mtz'", default="friedel_plus.mtz", type=str)
    parser.add_argument("-m", "--friedel-minus-mtz", help="Output mtz with Minus Friedel mates. Default 'friedel_minus.mtz'", default="friedel_minus.mtz", type=str)
    return parser

def split_friedel(args=None):
    if args is None:
        parser = get_split_friedel_parser()
        args = parser.parse_args()

    ds = rs.read_mtz(args.unmerged_mtz)
    if ds.merged:
        raise ValueError(f"Expected an unmerged Mtz, but {args.unmerged_mtz} is merged.")

    # M/ISYM is part of the MTZ specification and can be used
    # to determine the sign of a Friedel mate. More info at:
    # https://www.ccp4.ac.uk/html/mtzformat.html#column-labels-and-standard-names
    plus = (ds.hkl_to_asu()["M/ISYM"].to_numpy() % 2 == 1)

    #The double-Wilson prior expects all the centrics in the friedel plus file
    centrics = ds.label_centrics().CENTRIC.to_numpy()
    plus_or_centric = plus | centrics

    ds[plus_or_centric].write_mtz(args.friedel_plus_mtz)
    ds[~plus_or_centric].write_mtz(args.friedel_minus_mtz)


def get_combine_friedel_parser():
    parser = ArgumentParser(__doc__)
    parser.add_argument("plus_mtz")
    parser.add_argument("minus_mtz")
    parser.add_argument("out_mtz")
    return parser

def combine_friedel(args=None):
    if args is None:
        parser = get_combine_friedel_parser()
        args = parser.parse_args()

    plus = rs.read_mtz(args.plus_mtz)
    minus = rs.read_mtz(args.minus_mtz)

    is_xval_mtz = False
    if ('repeat' in plus) or ('half' in plus):
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
    if is_xval_mtz:
        group_keys = ['half', 'repeat']
        cell,sg = out.cell,out.spacegroup
        out = out.groupby(group_keys).apply(
            lambda x: x.drop(columns=group_keys).unstack_anomalous()[anom_keys]
        )
        out.cell,out.spacegroup = cell,sg
    else:
        out = out.unstack_anomalous()[anom_keys]

    out['F(+)']=out['F(+)'].astype("G")
    out['F(-)']=out['F(-)'].astype("G")
    out['SigF(+)']=out['SigF(+)'].astype("L")
    out['SigF(-)']=out['SigF(-)'].astype("L")

    out.write_mtz(args.out_mtz)

