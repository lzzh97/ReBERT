"""Step 4: group bits into words from the score matrices and compare with the ground truth.

Reports, per benchmark and replacement index: fully / partially / not found
words (%) and ARI, NMI, AMI. Predicted words go to work/grouping_res/ and the
table to work/results_<tag>.csv.

Threshold: two bits are linked when their score exceeds it.
  --threshold auto   max score / 3 for each matrix (default)
  --threshold 1.5    a fixed value
  --sweep            try 0.1, 0.2, ..., 2.9 and keep the best ARI. This picks the
                     threshold using the ground truth, so it is an upper bound.
"""
import argparse
import os

import numpy as np
import pandas as pd

from rebert.grouping import grouping_bits, score_grouping, write_groups
from rebert.netlist import read_words
from rebert.paths import BENCHES, DEFAULT_BENCH_DIR, DEFAULT_WORK_DIR, LEVEL, REP_IDX, Paths, parse_rep_idx


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tag', default='full', help='tag used by predict.py')
    p.add_argument('--benches', nargs='+', default=BENCHES)
    p.add_argument('--rep-idx', nargs='+', default=[str(r) for r in REP_IDX])
    p.add_argument('--level', type=int, default=LEVEL)
    p.add_argument('--threshold', default='auto')
    p.add_argument('--sweep', action='store_true')
    p.add_argument('--bench-dir', default=DEFAULT_BENCH_DIR)
    p.add_argument('--work-dir', default=DEFAULT_WORK_DIR)
    args = p.parse_args()

    paths = Paths(args.bench_dir, args.work_dir)
    os.makedirs(paths.grouping_dir, exist_ok=True)

    rows = []
    for bench in args.benches:
        truth = read_words(paths.words(bench))
        for rep_idx in parse_rep_idx(args.rep_idx):
            matrix = pd.read_csv(paths.score_matrix(args.tag, bench, args.level, rep_idx), index_col=0)
            if args.sweep:
                candidates = np.round(np.arange(0.1, 3, 0.1), 1)
            elif args.threshold == 'auto':
                candidates = [matrix.values.max() / 3]
            else:
                candidates = [float(args.threshold)]

            best = None
            for th in candidates:
                groups = grouping_bits(matrix, threshold=th)
                res = score_grouping(groups, truth)
                if best is None or res['ari'] > best[1]['ari']:
                    best = (th, res, groups)

            th, res, groups = best
            write_groups(groups, paths.grouping(args.tag, bench, rep_idx))
            rows.append(dict(bench=bench, rep_idx=rep_idx, threshold=th, **res))
            print(f"{bench} rep_idx={rep_idx} th={th:.2f}  ARI={res['ari']:.3f}  "
                  f"fully={res['fully_found']:.1f}%  not={res['not_found']:.1f}%")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(paths.work_dir, f'results_{args.tag}.csv')
    df.to_csv(out_csv, index=False)
    print('\nMean over benchmarks, per replacement index:')
    print(df.groupby('rep_idx')[['fully_found', 'partially_found', 'not_found', 'ari', 'nmi', 'ami']].mean().round(3))
    print(f'\nsaved {out_csv}')


if __name__ == '__main__':
    main()
