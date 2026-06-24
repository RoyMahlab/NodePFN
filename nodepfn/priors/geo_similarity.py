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
              single_eval_pos=None, batch_size_per_gp_sample=None, **kwargs):
    """Draw one geometric-similarity graph and return (x, y, target_y, edge_index).

    x: (T, B, num_features)   y, target_y: (T, B)   edge_index: (2, E) over T nodes.
    """
    hyperparameters = hyperparameters or {}
    # Optional pins for the sampled config, e.g. {'similarity': 'cosine'}.
    fixed = dict(hyperparameters.get('geo_fixed_hparams', {}) or {})
    rotate = hyperparameters.get('rotate_normalized_labels', True)
    # The model's classification head has exactly max_num_classes outputs and its
    # CrossEntropyLoss weight tensor is the same width, so a graph that produced more
    # classes would push targets out of bounds. sample_config draws n_classes up to 20;
    # cap it at the head width (n_classes does not affect GraphConfig.is_valid()).
    max_num_classes = hyperparameters.get('max_num_classes')
    # Topology-safe analogue of flexible_categorical's check_is_compatible: ensure the
    # context split [:single_eval_pos] and the query split [single_eval_pos:] expose the
    # same (>1) classes, so every queried class has in-context evidence. We cannot permute
    # nodes to fix a bad split (the SCM topology in edge_index is tied to node positions and
    # is shared across the batch), so we regenerate the whole graph instead. Off by default.
    
    # check_compat = hyperparameters.get('check_is_compatible', False) and single_eval_pos
    check_compat = True
    max_retries = int(hyperparameters.get('compatible_max_retries', 10))

    def _is_compatible(y_raw):
        train_classes = torch.unique(y_raw[:single_eval_pos])
        eval_classes = torch.unique(y_raw[single_eval_pos:])
        return (train_classes.numel() > 1
                and train_classes.numel() == eval_classes.numel()
                and bool((train_classes == eval_classes).all()))

    # One consistent graph with exactly seq_len nodes. Seed the hyperparameter stream from
    # the (globally seeded) torch RNG so the run honours set_seed while still varying
    # graph-to-graph; the graph internals use the torch/np globals.
    cfg = data = None
    for _ in range(max_retries if check_compat else 1):
        seed = int(torch.randint(0, 2 ** 31 - 1, (1,)).item())
        rng = np.random.default_rng(seed)
        cfg = sample_config(rng=rng, n_nodes=int(seq_len), **fixed)
        if max_num_classes is not None:
            cfg.n_classes = min(cfg.n_classes, int(max_num_classes))
        data = CausalGraphGenerator(cfg).generate()
        if not check_compat or _is_compatible(data['y']):
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
              f'sim={cfg.similarity} tau={cfg.sim_threshold:.2f} frame={cfg.normalize}')

    return x, y, y, edge_index


DataLoader = get_batch_to_dataloader(get_batch)
