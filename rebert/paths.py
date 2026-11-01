"""Where inputs live and where every pipeline stage writes its outputs."""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BENCH_DIR = os.path.join(REPO_ROOT, 'benchmarks', 'itc99')
DEFAULT_WORK_DIR = os.path.join(REPO_ROOT, 'work')

# Benchmarks and gate replacement indices used in the paper
BENCHES = ['b03', 'b04', 'b05', 'b07', 'b08', 'b11', 'b12', 'b13', 'b14', 'b15', 'b17', 'b18']
REP_IDX = [0, 0.2, 0.4, 0.6, 0.8, 1]
LEVEL = 6


class Paths:
    def __init__(self, bench_dir=DEFAULT_BENCH_DIR, work_dir=DEFAULT_WORK_DIR):
        self.bench_dir = bench_dir
        self.work_dir = work_dir
        self.twoinputs_dir = os.path.join(work_dir, 'twoinputs')
        self.tree_dir = os.path.join(work_dir, 'treegraph')
        self.pair_dir = os.path.join(work_dir, 'pair_data')
        self.score_dir = os.path.join(work_dir, 'score_matrix')
        self.grouping_dir = os.path.join(work_dir, 'grouping_res')
        self.ckpt_dir = os.path.join(work_dir, 'ckpts')

    def bench(self, bench):
        return os.path.join(self.bench_dir, bench, f'{bench}.bench')

    def words(self, bench):
        return os.path.join(self.bench_dir, bench, f'{bench}.words')

    def twoinputs(self, bench, rep_idx):
        return os.path.join(self.twoinputs_dir, f'{bench}_repidx{rep_idx}_twoinputs.txt')

    def tree(self, bench, level, rep_idx, bit):
        return os.path.join(self.tree_dir, f'{bench}_{level}level_repidx{rep_idx}_{bit}.json')

    def pairs(self, bench):
        return os.path.join(self.pair_dir, f'{bench}.pkl')

    def score_matrix(self, tag, bench, level, rep_idx):
        return os.path.join(self.score_dir, f'{tag}_{bench}_lv{level}_idx{rep_idx}.csv')

    def grouping(self, tag, bench, rep_idx):
        return os.path.join(self.grouping_dir, f'{tag}_{bench}_repidx{rep_idx}.txt')


def parse_rep_idx(values):
    """'0 0.2 1' -> [0, 0.2, 1], keeping ints as ints so file names match the paper runs."""
    out = []
    for v in values:
        f = float(v)
        out.append(int(f) if f.is_integer() else f)
    return out
