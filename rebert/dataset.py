"""Labeled bit pairs and the torch Dataset that feeds them to ReBERT."""
import itertools
import os
import random

import torch
from torch.utils.data import Dataset

from rebert.netlist import netlist2expressions, read_words
from rebert.paths import Paths
from rebert.tree import (TREE_PE_DIM, load_graph, parse_expression, save_graph,
                         tree_positional_embedding, tree_to_text)

MAX_POSITIVE_PAIRS = 300


def build_trees(paths, bench, level, rep_idx):
    """Write one expression-tree JSON per target bit of a (gate-replaced) netlist."""
    with open(paths.twoinputs(bench, rep_idx), 'r') as file:
        netlist_content = file.read()
    expressions = netlist2expressions(netlist_content, level, paths.words(bench))
    os.makedirs(paths.tree_dir, exist_ok=True)
    for bit, expr in expressions[f"{level}_level_expr_x"].items():
        save_graph(parse_expression(expr), paths.tree(bench, level, rep_idx, bit))


def get_labeled_data(paths, bench, level, rep_idx, balance=None, rng=random):
    """All bit pairs of a benchmark as (bench, level, bit1, bit2, rep_idx, label).

    label is 1 when both bits are in the same ground-truth word. With `balance`,
    keep at most MAX_POSITIVE_PAIRS positives and `balance` times as many negatives.
    """
    item_to_group = {}
    for group_id, group in enumerate(read_words(paths.words(bench))):
        for item in group:
            item_to_group[item] = group_id

    pair_labels = []
    for item1, item2 in itertools.combinations(item_to_group, 2):
        label = 1 if item_to_group[item1] == item_to_group[item2] else 0
        pair_labels.append((bench, level, item1, item2, rep_idx, label))

    if balance is None:
        return pair_labels

    positive_labels = [item for item in pair_labels if item[5] == 1]
    negative_labels = [item for item in pair_labels if item[5] == 0]
    if len(positive_labels) > MAX_POSITIVE_PAIRS:
        min_count = MAX_POSITIVE_PAIRS
    else:
        min_count = min(len(positive_labels), len(negative_labels))
    n_negative = min(int(min_count * balance), len(negative_labels))
    return rng.sample(positive_labels, min_count) + rng.sample(negative_labels, n_negative)


def pad_pe(pe, bert_hidden_size):
    current_dim = pe.shape[1]
    if current_dim > bert_hidden_size:
        raise ValueError(f"Positional encoding dimension {current_dim} exceeds BERT hidden size {bert_hidden_size}.")
    if current_dim < bert_hidden_size:
        pe = torch.cat((pe, torch.zeros((pe.size(0), bert_hidden_size - current_dim))), dim=1)
    return pe


class BitPairDataset(Dataset):
    def __init__(self, data, paths, max_len=512, bert_hidden_size=768):
        assert TREE_PE_DIM <= bert_hidden_size
        self.data = data
        self.paths = paths
        self.bert_hidden_size = bert_hidden_size
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        bench, level, item1, item2, rep_idx, label = self.data[idx]
        G1 = load_graph(self.paths.tree(bench, level, rep_idx, item1))
        G2 = load_graph(self.paths.tree(bench, level, rep_idx, item2))
        pe1 = pad_pe(tree_positional_embedding(G1), self.bert_hidden_size)
        pe2 = pad_pe(tree_positional_embedding(G2), self.bert_hidden_size)

        # Line up with the tokenizer output: [CLS] tree1 [SEP] tree2 [SEP]
        zero_pe = torch.zeros((1, self.bert_hidden_size))
        pe = torch.cat((zero_pe, pe1, zero_pe, pe2, zero_pe), dim=0)
        if pe.size(0) < self.max_len:
            pe = torch.cat((pe, torch.zeros((self.max_len - pe.size(0), self.bert_hidden_size))), dim=0)
        else:
            pe = pe[:self.max_len, :]
        return {
            'text1': tree_to_text(G1),
            'text2': tree_to_text(G2),
            'pe': pe,
            'labels': label,
        }


def collate(batch):
    return {
        'text1': [item['text1'] for item in batch],
        'text2': [item['text2'] for item in batch],
        'pe': torch.stack([item['pe'] for item in batch]),
        'labels': torch.tensor([item['labels'] for item in batch]),
    }
