"""Score the structural matching baseline outputs in baselines/structural_matching/.

Each file <bench>_rep<r>.txt lists predicted words as `{ a, b, c, }`, one per line.
Prints the same metrics as evaluate.py and writes work/results_baseline.csv.
"""
import argparse
import glob
import os
import re

import pandas as pd

from rebert.grouping import score_grouping
from rebert.netlist import read_words
from rebert.paths import DEFAULT_BENCH_DIR, DEFAULT_WORK_DIR, REPO_ROOT, Paths

DEFAULT_BASELINE_DIR = os.path.join(REPO_ROOT, 'baselines', 'structural_matching')


def read_brace_groups(path):
    groups = []
    with open(path, 'r') as f:
        for line in f:
            inner = re.search(r'\{([^{}]*)\}', line)
            if inner:
                bits = [b.strip() for b in inner.group(1).split(',') if b.strip()]
                if bits:
                    groups.append(bits)
    return groups


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--baseline-dir', default=DEFAULT_BASELINE_DIR)
    p.add_argument('--bench-dir', default=DEFAULT_BENCH_DIR)
    p.add_argument('--work-dir', default=DEFAULT_WORK_DIR)
    args = p.parse_args()

    paths = Paths(args.bench_dir, args.work_dir)
    rows = []
    for path in sorted(glob.glob(os.path.join(args.baseline_dir, 'b*_rep*.txt'))):
        bench, rep_idx = re.match(r'(b\d+)_rep([\d.]+)\.txt', os.path.basename(path)).groups()
        res = score_grouping(read_brace_groups(path), read_words(paths.words(bench)))
        rows.append(dict(bench=bench, rep_idx=float(rep_idx), **res))

    df = pd.DataFrame(rows).sort_values(['bench', 'rep_idx'])
    print(df.round(3).to_string(index=False))
    os.makedirs(paths.work_dir, exist_ok=True)
    out_csv = os.path.join(paths.work_dir, 'results_baseline.csv')
    df.to_csv(out_csv, index=False)
    print(f'\nsaved {out_csv}')


if __name__ == '__main__':
    main()
