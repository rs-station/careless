"""
Summary statistics for data reduction. 
"""
import argparse
import matplotlib.pyplot as plt
import reciprocalspaceship as rs
import pandas as pd
import seaborn as sns
import numpy as np
import glob
import re
import copy

from careless.stats.parser import BaseParser
from careless.stats.completeness import run_analysis as run_completeness
from careless.stats.completeness import ArgumentParser as compl_parser
from careless.stats.cchalf import run_analysis as run_cchalf
from careless.stats.cchalf import ArgumentParser as cchalf_parser
from careless.stats.isigi import run_analysis as run_isigi
from careless.stats.isigi import ArgumentParser as isigi_parser

class ArgumentParser(BaseParser):
    def __init__(self):
        super().__init__(
            description=__doc__
        )

        # Required arguments
        self.add_argument(
            "careless_dir",
            help="careless output dir containing merged MTZs and xval MTZs",
        )
        self.add_argument(
            "in_mtzs",             
            nargs="+",
            help="input unmerged mtz files in the order of careless outputs",
        )
        self.add_argument(
            "-b",
            "--bins",
            default=10,
            type=int,
            help=("Number of resolution bins to use, the default is 10."),
        )
        self.add_argument(
            "-B",
            "--batch_key",
            default=None,
            type=str,
            help=("BATCH column name"),
        )


def _get_various_stats(unmerged_dir, merged_dir, dmin = None, dmax = None, batch_key = None):
    out = {}
    unmerged_mtz = rs.read_mtz(unmerged_dir)
    merged_mtz = rs.read_mtz(merged_dir)
    unmerged_mtz = unmerged_mtz.compute_dHKL().query(f"dHKL >= {dmin}")
    n_refs = len(unmerged_mtz)
    n_refs_merged = len(merged_mtz.compute_dHKL().query(f"dHKL >= {dmin}", engine="python"))
    unmerged_hr = unmerged_mtz.query(f"dHKL >= {dmin} and dHKL <= {dmax}", engine="python")
    n_refs_hr = len(unmerged_hr)
    n_refs_hr_merged = len(merged_mtz.compute_dHKL().query(f"dHKL >= {dmin} and dHKL <= {dmax}", engine="python"))

    if batch_key:
        try:
            out["# of Images"] = len(unmerged_mtz[batch_key].value_counts())
        except:
            raise ValueError("incorrect BATCH key!")
        
    mult_o = n_refs/n_refs_merged
    mult_hr = n_refs_hr/n_refs_hr_merged

    cell = merged_mtz.cell
    out["Space group"] = merged_mtz.spacegroup.hm 
    out["Cell constants"] = ""
    out["   a (Å)"] = cell.a
    out["   b (Å)"] = cell.b
    out["   c (Å)"] = cell.c
    out["   α (°)"] = cell.alpha
    out["   β (°)"] = cell.beta
    out["   γ (°)"] = cell.gamma
    out["Total Obs."] = f"{n_refs} ({n_refs_hr})"
    out["Unique Obs."] = f"{n_refs_merged} ({n_refs_hr_merged})"
    out["Multiplicity"] = f"{mult_o:0.1f} ({mult_hr:0.1f})"
    
    return out
    
def run_analysis(args):
    
    in_mtzs = args.in_mtzs
    n_out = len(in_mtzs)
    merged_mtzs = [
        p for p in glob.glob(args.careless_dir+"*_*.mtz")
        if re.match(r"^(?!.*(?:_predictions|_xval)_\d+\.mtz$).+_\d+\.mtz$", p)
    ]
    merged_mtzs = sorted(merged_mtzs)

    xval_mtzs = [
        p for p in  glob.glob(args.careless_dir+"*_xval_*.mtz")
    ]
    xval_mtzs = sorted(xval_mtzs)

    if not (len(xval_mtzs) == len(merged_mtzs) == n_out):
        raise ValueError(f"""unequal numbers of unmerged, merged, and xval MTZ files:
        unmerged MTZs: {in_mtzs}, 
        _____________________________________________
        merged MTZs: {merged_mtzs}, 
        _____________________________________________
        xval MTZs: {xval_mtzs}
        """)

    completeness_results = {}
    for mtz in merged_mtzs:
        compl_args = compl_parser().parse_args([mtz, f"-b {args.bins}"])
        compl = run_completeness(compl_args)
        completeness_results[mtz] = compl
    print(completeness_results)
    cchalf_args = cchalf_parser().parse_args([*xval_mtzs, f"-b {args.bins}"])
    cchalf_results = run_cchalf(cchalf_args)

    cchalf_overall_args = cchalf_parser().parse_args([*xval_mtzs, f"-b 1"])
    cchalf_overall_results = run_cchalf(cchalf_overall_args)

    isigi_args = isigi_parser().parse_args([*merged_mtzs, f"-b {args.bins}"])
    isigi_results = run_isigi(isigi_args)

    isigi_overall_args = isigi_parser().parse_args([*merged_mtzs, f"-b 1"])
    isigi_overall_results = run_isigi(isigi_overall_args)

    out = []
    for ds_num in range(n_out):
        stats = {}
        stats["in_mtz"] = in_mtzs[ds_num]
        stats["shortname"] = re.split("/",stats["in_mtz"])[-1]

        compl_overall = (
            completeness_results[merged_mtzs[0]]
            .query("`Resolution Range (Å)` == 'overall'")
            ["non-anomalous"]
            .iloc[0]
        )
        compl_lastbin = (
            completeness_results[merged_mtzs[0]]
            ["non-anomalous"]
            .iloc[-1]
        )
        stats["Completeness"] = f"{compl_overall:0.3f} ({compl_lastbin:0.3f})"

        isigi_input = re.split("/",merged_mtzs[ds_num])[-1]
        isigi_overall = isigi_overall_results.query(f"file == '{isigi_input}'")["I/sigI"].iloc[0]
        isigi_lastbin = isigi_results.query(f"file == '{isigi_input}'")["I/sigI"].iloc[-1]
        stats["I/sigI"] = f"{isigi_overall:0.3f} ({isigi_lastbin:0.3f})"

        res_range_overall = isigi_overall_results.query(
            f"file == '{isigi_input}'")["Resolution Range (Å)"].iloc[0]
        res_range_lastbin = isigi_results.query(f"file == '{isigi_input}'")["Resolution Range (Å)"].iloc[-1]
        dmin_last = re.split(r'\s+',res_range_lastbin)[2]
        dmax_last = re.split(r'\s+',res_range_lastbin)[0]
        stats["Resolution Range (Å)"] = f"{res_range_overall} ({res_range_lastbin})"
        stats = {"Resolution Range (Å)": stats.pop("Resolution Range (Å)"), **stats}

        
        cchalf_input = xval_mtzs[ds_num]
        cchalf_overall = cchalf_overall_results.query(f"file == '{cchalf_input}'")["CChalf"]
        cchalf_overall = cchalf_overall.mean()
        cchalf_lastbin = cchalf_results.query(f"file == '{cchalf_input}'")["CChalf"].iloc[-1]
        cchalf_lastbin = cchalf_lastbin.mean()
        stats["CC1/2"] = f"{cchalf_overall:0.3f} ({cchalf_lastbin:0.3f})"

        simple_stats = _get_various_stats(in_mtzs[ds_num], merged_mtzs[ds_num], dmin_last, dmax_last, batch_key=args.batch_key)
        stats = simple_stats|stats
        out.append(stats)

        results = pd.DataFrame(out)
        target = ['in_mtz', "shortname"]
        all_other_cols = [col for col in results.columns if col not in target]

        results = results[target + all_other_cols].T
        print(results)
        if args.output is not None:
            results.to_csv(args.output)
        else:
            print(results)
    return results

    #out = pd.concat(out)
    #return out
def main():
    parser = ArgumentParser().parse_args()
    print(parser.output)
    run_analysis(parser)
    print("done")