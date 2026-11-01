"""Fast checks of the data pipeline on the smallest benchmark (no model download needed)."""
import os
import random

import pandas as pd

from rebert.dataset import BitPairDataset, build_trees, get_labeled_data
from rebert.gate_replace import replace_gates
from rebert.grouping import grouping_bits, score_grouping
from rebert.netlist import netlist2expressions, parse_netlist, read_words
from rebert.paths import Paths
from rebert.tree import parse_expression, preorder_traversal, tree_positional_embedding, tree_to_text

BENCH = 'b03'


def test_parse_netlist():
    netlist = 'INPUT(A)\nQ = DFF(D)\nD = NAND(A, B)\nB = NOT(A)\n'
    assert parse_netlist(netlist) == {'Q': 'D', 'D': 'NAND(A,B)', 'B': 'NOT(A)'}


def test_tree_roundtrip():
    G = parse_expression('NAND(X[0],AND(X[1],X[2]))')
    assert preorder_traversal(G) == ['NAND', 'X[0]', 'AND', 'X[1]', 'X[2]']
    assert tree_to_text(G) == 'NAND X AND X X'
    pe = tree_positional_embedding(G)
    assert pe.shape == (5, 512)
    assert pe[0].sum() == 0


def test_replace_gates_keeps_two_inputs(tmp_path):
    paths = Paths(work_dir=str(tmp_path))
    for ratio in (0, 0.5, 1):
        out = tmp_path / f'r{ratio}.txt'
        replace_gates(paths.bench(BENCH), str(out), ratio)
        for line in out.read_text().splitlines():
            if '(' in line and '=' in line:
                assert line.count(',') <= 1, line
    # Full replacement leaves no NAND/OR/NOR gates
    assert 'NAND(' not in (tmp_path / 'r1.txt').read_text()


def test_end_to_end_without_model(tmp_path):
    paths = Paths(work_dir=str(tmp_path))
    os.makedirs(paths.twoinputs_dir)
    replace_gates(paths.bench(BENCH), paths.twoinputs(BENCH, 0.2), 0.2)
    build_trees(paths, BENCH, 6, 0.2)

    with open(paths.twoinputs(BENCH, 0.2)) as f:
        exprs = netlist2expressions(f.read(), 6, paths.words(BENCH))
    truth = read_words(paths.words(BENCH))
    assert len(exprs) == sum(len(w) for w in truth)

    pairs = get_labeled_data(paths, BENCH, 6, 0.2, balance=1.2, rng=random.Random(0))
    item = BitPairDataset(pairs, paths)[0]
    assert item['pe'].shape == (512, 768)

    # Perfect scores must reproduce the ground truth exactly
    bits = sorted(b for w in truth for b in w)
    same = {(a, b) for w in truth for a in w for b in w}
    matrix = pd.DataFrame([[1.0 if (a, b) in same else 0.0 for b in bits] for a in bits], index=bits, columns=bits)
    res = score_grouping(grouping_bits(matrix, threshold=0.5), truth)
    assert res['fully_found'] == 100 and abs(res['ari'] - 1) < 1e-9
