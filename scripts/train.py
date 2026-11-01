"""Step 2: train the ReBERT pair classifier on the pairs written by prepare_data.py.

The final model is saved to work/ckpts/<variant>_lv<level>/final (model.safetensors).
"""
import argparse
import os
import pickle
import random

import numpy as np
import torch
from transformers import Trainer, TrainingArguments

from rebert.dataset import BitPairDataset, collate
from rebert.model import VARIANTS, ReBERT
from rebert.paths import BENCHES, DEFAULT_BENCH_DIR, DEFAULT_WORK_DIR, LEVEL, Paths


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--variant', choices=list(VARIANTS), default='full',
                   help='full: word+seq+tree embedding (paper); no_tree / tree_only: ablations')
    p.add_argument('--benches', nargs='+', default=BENCHES)
    p.add_argument('--exclude', nargs='*', default=[], help='benchmarks to hold out from training')
    p.add_argument('--level', type=int, default=LEVEL)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--eval-frac', type=float, default=0.1, help='fraction of pairs used for validation')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--bench-dir', default=DEFAULT_BENCH_DIR)
    p.add_argument('--work-dir', default=DEFAULT_WORK_DIR)
    p.add_argument('--output', help='checkpoint dir (default: work/ckpts/<variant>_lv<level>)')
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    paths = Paths(args.bench_dir, args.work_dir)
    output = args.output or os.path.join(paths.ckpt_dir, f'{args.variant}_lv{args.level}')

    pair_data = []
    for bench in args.benches:
        if bench in args.exclude:
            continue
        with open(paths.pairs(bench), 'rb') as f:
            pair_data += [d for d in pickle.load(f) if d[1] == args.level]
    random.shuffle(pair_data)
    n_train = int((1 - args.eval_frac) * len(pair_data))
    print(f'{len(pair_data)} pairs: {n_train} train / {len(pair_data) - n_train} eval')
    train_dataset = BitPairDataset(pair_data[:n_train], paths)
    eval_dataset = BitPairDataset(pair_data[n_train:], paths)

    model = ReBERT(args.variant)

    training_args = TrainingArguments(
        output_dir=output,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        seed=args.seed,
        report_to='none',
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate,
    )
    trainer.train()
    print(trainer.evaluate())
    trainer.save_model(os.path.join(output, 'final'))
    print(f'saved to {os.path.join(output, "final")}')


if __name__ == '__main__':
    main()
