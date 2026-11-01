"""Step 1: gate replacement, expression trees and labeled training pairs.

For each benchmark and gate replacement index this writes
  work/twoinputs/<bench>_repidx<r>_twoinputs.txt   corrupted 2-input netlist
  work/treegraph/<bench>_<level>level_repidx<r>_<bit>.json   one tree per target bit
and per benchmark
  work/pair_data/<bench>.pkl                       balanced labeled pairs for training
"""
import argparse
import os
import pickle
import random
import time

from rebert.dataset import build_trees, get_labeled_data
from rebert.gate_replace import replace_gates
from rebert.paths import BENCHES, DEFAULT_BENCH_DIR, DEFAULT_WORK_DIR, LEVEL, REP_IDX, Paths, parse_rep_idx


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--benches', nargs='+', default=BENCHES)
    p.add_argument('--rep-idx', nargs='+', default=[str(r) for r in REP_IDX], help='gate replacement indices in [0, 1]')
    p.add_argument('--level', type=int, default=LEVEL, help='fan-in cone depth')
    p.add_argument('--balance', type=float, default=1.2, help='negative/positive pair ratio for training pairs')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--bench-dir', default=DEFAULT_BENCH_DIR)
    p.add_argument('--work-dir', default=DEFAULT_WORK_DIR)
    args = p.parse_args()

    paths = Paths(args.bench_dir, args.work_dir)
    rep_idx_list = parse_rep_idx(args.rep_idx)
    rng = random.Random(args.seed)
    for d in (paths.twoinputs_dir, paths.tree_dir, paths.pair_dir):
        os.makedirs(d, exist_ok=True)

    for bench in args.benches:
        t0 = time.time()
        pair_data = []
        for rep_idx in rep_idx_list:
            replace_gates(paths.bench(bench), paths.twoinputs(bench, rep_idx), rep_idx, seed=args.seed)
            build_trees(paths, bench, args.level, rep_idx)
            pair_data += get_labeled_data(paths, bench, args.level, rep_idx, balance=args.balance, rng=rng)
        with open(paths.pairs(bench), 'wb') as f:
            pickle.dump(pair_data, f)
        print(f'{bench}: {len(pair_data)} training pairs ({time.time() - t0:.1f}s)')


if __name__ == '__main__':
    main()
