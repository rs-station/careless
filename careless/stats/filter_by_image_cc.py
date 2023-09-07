"""
Filter reflections by image based on CCpred.
"""
import re
import argparse
import numpy as np
import reciprocalspaceship as rs
import gemmi

import matplotlib.pyplot as plt
import seaborn as sns

from careless.io.formatter import get_first_key_of_dtype

class ArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(
            description=__doc__
        )

        # Required arguments
        self.add_argument(
            "mtzs",
            nargs="+",
            help="A mix of *_predictions_#.mtz files and unmerged input files from careless. "
                 "Note: the filenames will be used to interpret what they are. Do not "
                 "use renamed files. The unmerged files must be supplied in the same order"
                 "they were presented to careless.",
        )

        # Optional arguments
        self.add_argument(
            "-m",
            "--method",
            default="pearson",
            choices=["pearson", "spearman"],
            help="Method for computing correlation coefficient (spearman or pearson). "
            "The Pearson CC uses maximum-likelihood weights. Pearson is the default.",
        )

        self.add_argument(
            "-c",
            "--cc-cutoff",
            required=True,
            type=float,
            help="The correlation cutoff. A number between zero and 1."
        )

        self.add_argument(
            "-o",
            default=None,
            help="Output filename base.",
        )

def weighted_pearson_ccfunc(df, iobs='Iobs', ipred='Ipred', sigiobs='SigIobs'):
    x = df[iobs].to_numpy('float32')
    y = df[ipred].to_numpy('float32')
    w = np.reciprocal(np.square(df[sigiobs])).to_numpy('float32')
    return rs.utils.weighted_pearsonr(x, y, w)

def spearman_ccfunc(df, iobs='Iobs', ipred='Ipred'):
    return df[[iobs, ipred]].corr(method='spearman')[iobs][ipred]

def is_predictions_filename(filename : str) -> bool:
    rematch = re.match(r".+predictions_[0-9]+\.mtz$", filename)
    retval = rematch is not None
    return retval

def predictions_id(filename : str) -> int:
    return int(filename[:-4].split('_')[-1])

def run_analysis(args):
    try:
        data_mtzs = [f for f in args.mtzs if not is_predictions_filename(f)]
        predictions_mtzs = [f for f in args.mtzs if is_predictions_filename(f)]
        predictions_mtzs = sorted(predictions_mtzs, key=predictions_id)
    except:
        raise ValueError(
            "Unable to interpret provided filenames {args.mtzs}. Please ensure "
            "predictions files have not been renamed. Unmerged mtzs must be in the"
            " order provided during the original careless run."
        )

    ds = []
    for m in predictions_mtzs:
        _ds = rs.read_mtz(m)
        #non-isomorphism could lead to different resolution for each mtz
        #need to calculate dHKL before concatenating 
        _ds.compute_dHKL(inplace=True)
        _ds['file'] = m
        _ds['Spacegroup'] = _ds.spacegroup.xhm()
        ds.append(_ds)
    ds = rs.concat(ds, check_isomorphous=False)

    grouper = ds.groupby(["file", "image_id"])

    if args.method.lower() == "spearman":
        ccfunc = spearman_ccfunc
    elif args.method.lower() == "pearson":
        ccfunc = weighted_pearson_ccfunc

    result = grouper.apply(ccfunc)
    result = rs.DataSet({"CCpred" : result}).reset_index()
    result['file_id'] = grouper.first()['file_id'].to_numpy()
    result['asu_id'] = grouper.first()['asu_id'].to_numpy()
    result = result[['file', 'file_id', 'asu_id', 'image_id', 'CCpred']]

    for i,m in enumerate(data_mtzs):
        ds = rs.read_mtz(m)
        if args.o is None:
            out = m[:-4] + '_filtered.mtz'
        else:
            out = args.o + f'_{i}.mtz'

        batch_key = get_first_key_of_dtype(ds, 'B')
        image_id = ds.groupby(batch_key).ngroup().to_numpy()
        cc = result[result.file_id == i].iloc[image_id].CCpred.to_numpy()
        idx = (cc >= args.cc_cutoff)
        ds[idx].copy().write_mtz(out)

def main():
    parser = ArgumentParser().parse_args()
    run_analysis(parser)

