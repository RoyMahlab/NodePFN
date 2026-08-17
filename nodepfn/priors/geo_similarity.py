"""Geometric-similarity causal prior.

Generates node-classification datasets from the ``casual_graph_generation`` package: a
structural causal model jointly produces node features ``X``, labels ``y``, and the graph
topology ``A`` (nodes connected when their geometric embeddings are similar). This replaces
the MLP-prior + ``generate_edge_index`` (SBM/random) path: here features, labels, and edges
all come from one SCM, so the topology is causally consistent with the features.

Architecture note: the transformer applies a SINGLE ``edge_index`` over ``T = seq_len``
nodes, shared across the batch dimension ``B`` (see ``layer.py``). So each ``get_batch`` call
draws ONE consistent geometric graph and shares it across the batch — every batch column is
the same graph (features, labels, and edges all match). Diversity comes across training
steps, each of which samples a fresh graph and a fresh hyperparameter configuration.
"""
import os
import sys

import numpy as np
import torch

from utils import (
    default_device,
    normalize_data,
    normalize_by_used_features_f,
    remove_outliers,
)
from .utils import get_batch_to_dataloader

# Make the repo-root ``casual_graph_generation`` package importable when pretraining is
# launched from inside ``nodepfn/`` (the usual cwd for these scripts).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from casual_graph_generation import CausalGraphGenerator, sample_config  # noqa: E402


def _stratified_split_perm(y: torch.Tensor, sep: int) -> torch.Tensor:
    """Return a node permutation placing >=1 sample of every class in the context split
    [:sep], so the eval classes are a subset of the context classes (ICL-compatible).

    Unlike flexible_categorical (whose edge_index is already built from pre-permutation
    node ids by the time it repairs a split), the geo prior still holds the DENSE
    adjacency here, so the caller can relabel the topology consistently with
    ``A[perm][:, perm]`` -- see the repair branch in get_batch.

    Falls back to a partial reservation when the class count exceeds the context size
    (sep >= min_eval_pos >= max_num_classes makes that unreachable in practice).
    """
    buckets: dict[int, list[int]] = {}
    for i, c in enumerate(y.tolist()):
        buckets.setdefault(c, []).append(i)
    reserved = [idxs[0] for idxs in buckets.values()]     # one guaranteed context sample per class
    pool = [i for idxs in buckets.values() for i in idxs[1:]]
    if pool:
        pool = [pool[i] for i in torch.randperm(len(pool)).tolist()]
    need = sep - len(reserved)
    if need >= 0:
        perm = reserved + pool[:need] + pool[need:]
    else:
        perm = reserved[:sep] + reserved[sep:] + pool
    return torch.tensor(perm, dtype=torch.long, device=y.device)


def _normalize_labels(y: torch.Tensor, rotate: bool = True) -> torch.Tensor:
    """Remap labels to contiguous 0..C-1 over the classes actually present, optionally
    applying a random cyclic shift (matches flexible_categorical's label normalisation)."""
    present = torch.unique(y, sorted=True)
    remap = torch.zeros(int(present.max().item()) + 1, dtype=torch.long, device=y.device)
    remap[present] = torch.arange(present.numel(), device=y.device)
    y = remap[y]
    n_classes = present.numel()
    if rotate and n_classes > 1:
        shift = int(torch.randint(0, n_classes, (1,)).item())
        y = (y + shift) % n_classes
    return y


@torch.no_grad()
def get_batch(batch_size, seq_len, num_features, hyperparameters=None, device=default_device,
              single_eval_pos=None, config_path: str | None = None, batch_size_per_gp_sample=None, **kwargs):
    """Draw one geometric-similarity graph and return (x, y, target_y, edge_index).

    x: (T, B, num_features)   y, target_y: (T, B)   edge_index: (2, E) over T nodes.

    Relevant hyperparameters: ``stratify_p`` (probability of repairing an incompatible
    context/eval split in place instead of regenerating the graph; 0 = old behaviour),
    ``compatible_max_retries``, ``compat_mode``, ``max_num_classes``,
    ``rotate_normalized_labels``, ``geo_fixed_hparams``, ``verbose``.
    """
    hyperparameters = hyperparameters or {}
    # Optional pins for the sampled config, e.g. {'similarity': 'cosine'}.
    fixed = dict(hyperparameters.get('geo_fixed_hparams', {}) or {})
    rotate = hyperparameters.get('rotate_normalized_labels', True)
    # The model's classification head has exactly max_num_classes outputs and its
    # CrossEntropyLoss weight tensor is the same width, so a graph that produced more
    # classes would push targets out of bounds. We pass max_num_classes into sample_config
    # so its n_classes draw spans the full head width, and clamp afterwards as a safety net
    # (n_classes does not affect GraphConfig.is_valid()).
    max_num_classes = hyperparameters.get('max_num_classes')
    # Topology-safe analogue of flexible_categorical's check_is_compatible: ensure every
    # queried (eval) class also appears in the context split [:single_eval_pos], so it has
    # in-context evidence.
    #
    # An incompatible draw is REPAIRED with probability `stratify_p` (default 1.0) by
    # permuting nodes so the context holds >=1 sample of every class, and permuting the
    # dense adjacency the same way (`A[perm][:, perm]`) so the topology stays consistent --
    # the graph is still dense here, edge_index is only built at the end. With probability
    # 1 - stratify_p (and while retries remain) the whole graph is regenerated instead,
    # which is what this prior used to do unconditionally.
    #
    # Repairing rather than rejecting matters for the class prior: regeneration redraws
    # n_classes too, and since acceptance falls off with class count, rejection reweights
    # the configured n_classes distribution towards low cardinalities (measured: mean
    # realized classes 23 against a configured mean of 51, and ~7% of graphs accepted with
    # queried classes that never appear in the context). Repairing leaves the configured
    # distribution intact and costs one draw instead of ~3.3.
    #
    # 'subset' (default) matches flexible_categorical: eval classes must be a subset of context
    # classes. The old 'exact' criterion (identical class sets) is almost never satisfiable past
    # a handful of classes, so it silently dropped high-cardinality graphs - the same collapse we
    # fixed in the causal prior. Set compat_mode='exact' to restore the strict check.
    check_compat = True
    max_retries = int(hyperparameters.get('compatible_max_retries', 10))
    compat_mode = os.environ.get('NODEPFN_COMPAT', hyperparameters.get('compat_mode', 'subset'))
    # Probability of repairing an incompatible split in place instead of regenerating.
    # stratify_p=0 restores the old regenerate-only behaviour.
    stratify_p = float(os.environ.get('NODEPFN_STRATIFY_P',
                                      hyperparameters.get('stratify_p', 1.0)))
    can_stratify = single_eval_pos is not None and compat_mode != 'exact'

    def _is_compatible(y_raw):
        train_classes = torch.unique(y_raw[:single_eval_pos])
        eval_classes = torch.unique(y_raw[single_eval_pos:])
        if compat_mode == 'exact':
            return (train_classes.numel() > 1
                    and train_classes.numel() == eval_classes.numel()
                    and bool((train_classes == eval_classes).all()))
        # 'subset'/'stratify': every eval class must appear in context, and >1 class present.
        return (eval_classes.numel() > 1
                and bool(torch.isin(eval_classes, train_classes).all()))

    # One consistent graph with exactly seq_len nodes. Seed the hyperparameter stream from
    # the (globally seeded) torch RNG so the run honours set_seed while still varying
    # graph-to-graph; the graph internals use the torch/np globals.
    cfg = data = None
    stratified = False
    for _ in range(max_retries if check_compat else 1):
        seed = int(torch.randint(0, 2 ** 31 - 1, (1,)).item())
        rng = np.random.default_rng(seed)
        cfg = sample_config(rng=rng, n_nodes=int(seq_len),
                            max_num_classes=max_num_classes, config_path=config_path, **fixed)
        if max_num_classes is not None:
            cfg.n_classes = min(cfg.n_classes, int(max_num_classes))
        data = CausalGraphGenerator(cfg).generate()
        if not check_compat or _is_compatible(data['y']):
            break
        if can_stratify and float(torch.rand(1).item()) < stratify_p:
            # Repair in place: reserve one node per class for the context split and relabel
            # nodes, features and the dense adjacency with the same permutation.
            perm = _stratified_split_perm(data['y'], int(single_eval_pos))
            data = {**data, 'X': data['X'][perm], 'y': data['y'][perm],
                    'A': data['A'][perm][:, perm]}
            if _is_compatible(data['y']):     # only unreachable when n_classes > single_eval_pos
                stratified = True
                break
    else:
        if hyperparameters.get('verbose'):
            print(f'[geo_similarity] no compatible split after {max_retries} draws; '
                  f'accepting last graph (single_eval_pos={single_eval_pos}).')

    X = data['X'].to(device).float()        # (T, n_feat)
    y = data['y'].to(device).long()         # (T,)
    A = data['A']                           # (T, T) dense 0/1 adjacency

    n_feat = X.shape[1]
    if n_feat > num_features:               # never expected (n_features <= max_features), but be safe
        X = X[:, :num_features]
        n_feat = num_features

    # Features -> (T, 1, H): outlier-clip, standardise, pad to num_features, scale by used fraction.
    x = X.unsqueeze(1)
    x = remove_outliers(x)
    x = normalize_data(x)
    if num_features > n_feat:
        pad = torch.zeros(x.shape[0], 1, num_features - n_feat, device=device)
        x = torch.cat([x, pad], dim=-1)
    x = normalize_by_used_features_f(x, n_feat, num_features)

    y = _normalize_labels(y, rotate=rotate).unsqueeze(1).float()   # (T, 1)

    # Share the single graph across the batch (architecture uses one edge_index over T nodes).
    x = x.repeat(1, batch_size, 1).contiguous()    # (T, B, H)
    y = y.repeat(1, batch_size).contiguous()       # (T, B)

    edge_index = A.nonzero().t().contiguous().to(device)   # (2, E)

    if hyperparameters.get('verbose'):
        print(f'[geo_similarity] T={seq_len} feat={n_feat}->{num_features} '
              f'classes={int(y.max().item()) + 1} edges={edge_index.shape[1]} '
              f'sim={cfg.similarity} tau={cfg.sim_threshold:.2f} frame={cfg.normalize} '
              f'stratified={stratified}')

    return x, y, y, edge_index


DataLoader = get_batch_to_dataloader(get_batch)
