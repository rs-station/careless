"Combine two half anomalous data sets into a single mtz"

import reciprocalspaceship as rs
from argparse import ArgumentParser

def get_parser():
    parser = ArgumentParser(__doc__)
    parser.add_argument("plus_mtz")
    parser.add_argument("minus_mtz")
    parser.add_argument("out_mtz")
    parser.add_argument("-x", "--is_xval", action="store_true")
    return parser

def unfriedelize(parser=None):
    if parser is None:
        parser = get_parser()
        parser = parser.parse_args()

    plus = rs.read_mtz(parser.plus_mtz)
    minus = rs.read_mtz(parser.minus_mtz)
    anom_keys = ['F(+)', 'SigF(+)', 'F(-)', 'SigF(-)', 'N(+)', 'N(-)']
    if parser.is_xval:
        anom_keys+=["repeat(+)","half(+)","repeat(-)","half(-)"]

    out = rs.concat([
        plus,
        minus.apply_symop("-x,-y,-z"),
    ]).unstack_anomalous()[anom_keys]

    if parser.is_xval:
        out = out[(out["half(+)"]==out["half(-)"])*(out["repeat(+)"]==out["repeat(-)"])]
        out = out.drop(columns=["repeat(-)","half(-)"])
        out = out.rename(columns={"repeat(+)":"repeat","half(+)":"half"})

    out['F(+)']=out['F(+)'].astype("G")
    out['F(-)']=out['F(-)'].astype("G")
    out['SigF(+)']=out['SigF(+)'].astype("L")
    out['SigF(-)']=out['SigF(-)'].astype("L")

    out.write_mtz(parser.out_mtz)

def main():
    return unfriedelize()

if __name__=="__main__":
    main()
