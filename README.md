# ReBERT

ReBERT is a BERT-based model that groups the bits of a gate-level netlist into words. It treats a netlist as a language: each bit's fan-in cone is written as a token sequence, and a tree-based positional embedding captures the structure of the circuit. A pair classifier predicts whether two bits belong to the same word, and a grouping step turns the pairwise scores into words.

Paper: **ReBERT: LLM for Gate-Level to Word-Level Reverse Engineering**, Lizi Zhang, Azadeh Davoodi, Rasit Onur Topaloglu. DATE 2025. [PDF](paper/ReBERT_DATE25.pdf)

On the ITC'99 benchmarks, ReBERT improves over a state-of-the-art structural matching method by 12.2% to 218.1% on average, depending on how heavily the netlist's structural patterns are corrupted.

## How it works

1. **Corruption (gate replacement).** A fraction *r* (the gate replacement index, 0 to 1) of the AND/NAND/OR/NOR gates is rewritten as AND-inverter logic, and every remaining wide gate is split into 2-input gates. This hides the structural patterns that matching methods rely on. The gates to rewrite are chosen at random (`--seed`), and this step is a reimplementation of the one used for the paper, so the numbers you get may differ slightly from the published ones.
2. **Tokenization.** For each target bit, its fan-in cone (6 levels by default) becomes an expression tree, serialized by pre-order traversal (`NAND X AND X X`). Two bits are joined as a BERT sentence pair.
3. **Embedding.** Word embedding + sequential position embedding + tree positional embedding (each node's path from the root, encoded as a 512-bit code).
4. **Pair prediction.** BERT (`bert-base-uncased`, fine-tuned) predicts whether the two bits belong to the same word.
5. **Word generation.** Bits whose pair score exceeds a threshold are linked; each connected component is a word.

## Repository layout

```
rebert/                    Python package
  netlist.py               .bench parsing, multi-level Boolean expressions
  gate_replace.py          gate replacement / 2-input splitting
  tree.py                  expression trees, pre-order tokens, tree positional embedding
  dataset.py               labeled bit pairs, torch Dataset
  model.py                 ReBERT model (full and ablation variants)
  grouping.py              pair scores -> words; fully/partially/not found, ARI/NMI/AMI
  paths.py                 benchmark list and output locations
scripts/
  prepare_data.py          step 1: corrupt netlists, build trees, sample training pairs
  train.py                 step 2: fine-tune the model
  predict.py               step 3: score all bit pairs of the test netlists
  evaluate.py              step 4: group bits into words and report metrics
  eval_baseline.py         score the structural matching baseline outputs
  run_all.sh               steps 1 to 4 in one go
benchmarks/itc99/<b>/      ITC'99 netlists (<b>.bench) and ground-truth words (<b>.words)
baselines/structural_matching/   baseline predictions, <bench>_rep<r>.txt
tests/                     quick checks of the data pipeline
paper/                     camera-ready paper
```

A `.words` file has one word per line, listing its bits separated by spaces. Every script writes into `work/` (ignored by git).

## Setup

Python 3.9 or newer. A CUDA GPU is strongly recommended for training.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # also installs this package in editable mode
pytest                           # about a minute; checks the data pipeline on b03
```

## Usage

Run everything with the paper's settings (training data from all 12 benchmarks, evaluation on b03 to b15):

```bash
bash scripts/run_all.sh
```

Or step by step:

```bash
# 1. Corrupted netlists, trees and training pairs for every benchmark and replacement index
python scripts/prepare_data.py

# 2. Train (variants: full = paper model, no_tree = without tree embedding, tree_only)
python scripts/train.py --variant full --epochs 20

# 3. Score bit pairs
python scripts/predict.py --variant full --ckpt work/ckpts/full_lv6/final --benches b03 b04 b05

# 4. Group into words and report metrics
python scripts/evaluate.py --tag full --benches b03 b04 b05
```

Each script accepts `--help`. Useful options:

- `--benches`, `--rep-idx`, `--level` select benchmarks, gate replacement indices (default `0 0.2 0.4 0.6 0.8 1`) and cone depth (default 6).
- `train.py --exclude b13` holds a benchmark out of training.
- `evaluate.py --threshold 1.5` fixes the grouping threshold; the default `auto` uses one third of the maximum score. `--sweep` reports the best threshold per netlist by ARI, which uses the ground truth and is therefore an upper bound.

To score the structural matching baseline on the same metrics:

```bash
python scripts/eval_baseline.py
```

## Metrics

For each ground-truth word: **fully found** if all its bits land in one predicted word, **not found** if no two of its bits share a predicted word, **partially found** otherwise. ARI, NMI and AMI compare the predicted and true partitions of the bits.

## Citation

```bibtex
@inproceedings{zhang2025rebert,
  title     = {ReBERT: LLM for Gate-Level to Word-Level Reverse Engineering},
  author    = {Zhang, Lizi and Davoodi, Azadeh and Topaloglu, Rasit Onur},
  booktitle = {Design, Automation and Test in Europe Conference (DATE)},
  year      = {2025}
}
```

## Acknowledgments

This work is supported by NSF Award No. 2322713. The benchmark circuits come from the ITC'99 suite (Politecnico di Torino).

## License

MIT, see [LICENSE](LICENSE).
