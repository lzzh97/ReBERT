"""Turn a pairwise score matrix into words, and score words against the ground truth."""
from itertools import combinations

import networkx as nx
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score


def write_groups(groups, save_path):
    with open(save_path, 'w') as f:
        for group in groups:
            f.write(' '.join(group) + '\n')


def read_groups(path):
    with open(path, 'r') as f:
        return [line.split() for line in f if line.strip()]


def grouping_bits(score_matrix, threshold=1, save_path=None):
    """Connect bits whose score exceeds `threshold`; each connected component is a word."""
    G = nx.Graph()
    G.add_nodes_from(score_matrix.index)
    for i in range(score_matrix.shape[0]):
        for j in range(i + 1, score_matrix.shape[1]):
            if score_matrix.iloc[i, j] > threshold:
                G.add_edge(score_matrix.index[i], score_matrix.columns[j])

    groups = [sorted(group) for group in nx.connected_components(G)]
    if save_path is not None:
        write_groups(groups, save_path)
    return groups


def agglomerative_grouping(distance_matrix, distance_threshold=0, save_path=None):
    """Alternative grouping: average-linkage clustering on a precomputed distance matrix."""
    clustering = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='average',
                                         distance_threshold=distance_threshold)
    clustering.fit(distance_matrix)
    grouped = {}
    for idx, label in enumerate(clustering.labels_):
        grouped.setdefault(label, []).append(distance_matrix.index[idx])
    groups = list(grouped.values())
    if save_path is not None:
        write_groups(groups, save_path)
    return groups


def _labels(groups):
    return {item: cid for cid, group in enumerate(groups) for item in group}


def similarity_metrics(predicted, ground_truth):
    """ARI, NMI and AMI over the bits present in both groupings."""
    pred_labels = _labels(predicted)
    true_labels = _labels(ground_truth)
    common_items = sorted(set(pred_labels) & set(true_labels))
    a = [pred_labels[item] for item in common_items]
    b = [true_labels[item] for item in common_items]
    return adjusted_rand_score(a, b), normalized_mutual_info_score(a, b), adjusted_mutual_info_score(a, b)


def has_any_pair_in_group(word, groups):
    for i in range(len(word)):
        for j in range(i + 1, len(word)):
            if any(word[i] in group and word[j] in group for group in groups):
                return True
    return False


def found_metrics(predicted, ground_truth):
    """Fractions of ground-truth words that are fully, partially and not found.

    fully found:     every bit of the word lands in one predicted word
    not found:       no two bits of the word share a predicted word
    partially found: everything else
    """
    pred_sets = [set(g) for g in predicted]
    fully, partially, not_found = 0, 0, 0
    for word in ground_truth:
        word_set = set(word)
        if any(word_set.issubset(g) for g in pred_sets):
            fully += 1
        elif not has_any_pair_in_group(list(word_set), pred_sets):
            not_found += 1
        else:
            partially += 1
    total = len(ground_truth)
    return fully / total, partially / total, not_found / total


def score_grouping(predicted, ground_truth):
    """All word-level metrics; found ratios are in percent."""
    ff, pf, nf = found_metrics(predicted, ground_truth)
    ari, nmi, ami = similarity_metrics(predicted, ground_truth)
    return dict(fully_found=ff * 100, partially_found=pf * 100, not_found=nf * 100, ari=ari, nmi=nmi, ami=ami)


def pair_accuracy(predicted, ground_truth):
    """Fraction of ground-truth bit pairs whose same-word / different-word status is predicted correctly."""
    true_groups = [set(g) for g in ground_truth]
    all_items = set().union(*true_groups)
    all_pairs = set(combinations(sorted(all_items), 2))
    true_pairs = set()
    for group in true_groups:
        true_pairs.update(combinations(sorted(group), 2))
    test_pairs = set()
    for group in predicted:
        test_pairs.update(combinations(sorted(group), 2))
    correct_pairs = (true_pairs & test_pairs) | ((all_pairs - true_pairs) & (all_pairs - test_pairs))
    return len(correct_pairs) / len(all_pairs) if all_pairs else 0
