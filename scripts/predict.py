"""Step 3: score every bit pair of the test netlists with a trained model.

Writes one symmetric score matrix per benchmark and replacement index to
work/score_matrix/<tag>_<bench>_lv<level>_idx<r>.csv. The score is the
"same word" logit. Pairs whose token sequences are too different
(Jaccard similarity below --jaccard) are not run through the model and get -1.
"""
import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from rebert.dataset import BitPairDataset, collate, get_labeled_data
from rebert.model import VARIANTS, ReBERT, load_checkpoint
from rebert.paths import BENCHES, DEFAULT_BENCH_DIR, DEFAULT_WORK_DIR, LEVEL, REP_IDX, Paths, parse_rep_idx


def jaccard_similarity(text1, text2):
    words1 = set(text1.split())
    words2 = set(text2.split())
    return len(words1 & words2) / len(words1 | words2)


@torch.no_grad()
def score_matrix(model, paths, bench, level, rep_idx, jaccard_min, device):
    pairs = get_labeled_data(paths, bench, level, rep_idx)
    loader = DataLoader(BitPairDataset(pairs, paths), batch_size=1, shuffle=False, collate_fn=collate)

    bit_names = sorted({p[2] for p in pairs} | {p[3] for p in pairs})
    matrix = pd.DataFrame(0.0, index=bit_names, columns=bit_names)
    cache = {}
    for pair, data in zip(pairs, loader):
        name1, name2 = pair[2], pair[3]
        text1, text2 = data['text1'][0], data['text2'][0]
        if jaccard_similarity(text1, text2) < jaccard_min:
            score = -1.0
        else:
            # Identical trees give identical inputs, so reuse their score
            key = f'{text1}::{text2}'
            if key not in cache:
                outputs = model(data['text1'], data['text2'], data['pe'].to(device))
                cache[key] = outputs['logits'][0, 1].item()
            score = cache[key]
        matrix.loc[name1, name2] = score
        matrix.loc[name2, name1] = score
    return matrix


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', required=True, help='directory with model.safetensors (e.g. work/ckpts/full_lv6/final)')
    p.add_argument('--variant', choices=list(VARIANTS), default='full')
    p.add_argument('--tag', help='name used in output files (default: the variant)')
    p.add_argument('--benches', nargs='+', default=BENCHES)
    p.add_argument('--rep-idx', nargs='+', default=[str(r) for r in REP_IDX])
    p.add_argument('--level', type=int, default=LEVEL)
    p.add_argument('--jaccard', type=float, default=0.7, help='minimum token Jaccard similarity to score a pair')
    p.add_argument('--bench-dir', default=DEFAULT_BENCH_DIR)
    p.add_argument('--work-dir', default=DEFAULT_WORK_DIR)
    args = p.parse_args()

    paths = Paths(args.bench_dir, args.work_dir)
    tag = args.tag or args.variant
    os.makedirs(paths.score_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_checkpoint(ReBERT(args.variant), args.ckpt).to(device).eval()

    for bench in args.benches:
        for rep_idx in parse_rep_idx(args.rep_idx):
            matrix = score_matrix(model, paths, bench, args.level, rep_idx, args.jaccard, device)
            out = paths.score_matrix(tag, bench, args.level, rep_idx)
            matrix.to_csv(out)
            print(f'{bench} rep_idx={rep_idx}: {len(matrix)} bits -> {out}')


if __name__ == '__main__':
    main()
