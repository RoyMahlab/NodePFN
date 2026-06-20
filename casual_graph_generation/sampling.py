"""Sample hyperparameter configurations matching the nodepfn causal prior.

The SCM-architecture and dataset hyperparameters are drawn from the SAME distributions
the nodepfn pretraining prior uses, so graphs generated here cover the same region of
hyperparameter space the model is pretrained on:

    casual_graph_generation   nodepfn source                              distribution
    -----------------------   -----------------------------------------   ---------------------------------------
    n_layers                  get_diff_causal: num_layers                 meta_gamma(max_alpha=2, max_scale=3,  lb=2)
    hidden                    get_diff_causal: prior_mlp_hidden_dim       meta_gamma(max_alpha=3, max_scale=100, lb=4)
    n_geo                     get_diff_causal: num_causes                 meta_gamma(max_alpha=3, max_scale=7,  lb=2)
    drop_rate                 get_diff_causal: prior_mlp_dropout_prob     meta_beta(scale=0.6, min=0.1, max=5.0)
    n_features                flexible_categorical: num_features_used     uniform_int(1, max_features=100)
    n_classes                 flexible_categorical: num_classes           constant = NUM_CLASSES (= max_num_classes = 10)

``n_classes`` is FIXED, not sampled: nodepfn's active config sets ``num_classes = 10`` (==
``max_num_classes``, the model's head width) and ``MulticlassRank(num_classes)`` discretises
the target into up to that many bins — the number of distinct classes actually present then
varies per dataset (empty bins). We mirror that: a constant ``n_classes`` fed to the SCM's
``discretise_labels``. (nodepfn's commented-out alternative was ``randint(2, 10)``; switch
``N_CLASSES_DIST`` below to ``{'distribution': 'uniform_int', 'min': 2, 'max': NUM_CLASSES}``
to reproduce that instead.)

The meta-distribution samplers below reproduce those in
``nodepfn/priors/differentiable_prior.py`` (hierarchical: first draw the meta-parameters
uniformly, then draw the value), using the per-call numpy ``Generator`` for reproducibility.

The similarity-kernel fields (``similarity``, ``sim_threshold``, ``normalize``,
``sim_out_dim``, ``sim_act``) and ``n_nodes`` are unique to this geometric-similarity
prior and have no nodepfn counterpart, so they keep their cell [23] dashboard ranges.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .constants import SYMMETRIC_ACTIVATIONS
from .generator import GraphConfig

# --- Choice sets from the cell [23] dropdown widgets (no nodepfn counterpart) ---
SIMILARITY_CHOICES = ['cosine', 'bilinear', 'mlp']
FRAME_CHOICES = ['activation', 'none', 'center', 'minmax', 'zscore', 'rank']
SIM_ACT_CHOICES = sorted(SYMMETRIC_ACTIVATIONS)

# --- Slider ranges from cell [23] for the geometric-similarity-only fields ---
# (field -> (min, max, step)); the rest now follow the nodepfn prior (PRIOR_DISTRIBUTIONS).
SLIDER_RANGES = {
    'n_nodes':     (16, 256, 8),   # = seq_len; not a per-graph SCM hyperparameter in nodepfn
    'sim_out_dim': (1, 32, 1),
}

# --- nodepfn scalar config values (model_configs.get_general_config /
#     get_flexible_categorical_config). These are NOT sampled in nodepfn; they bound the
#     sampled fields and fix the head width, so we mirror their exact values here. ---
MAX_FEATURES = 100      # get_general_config: max_features (upper bound of num_features_used)
MAX_NUM_CLASSES = 10    # get_flexible_categorical_config: max_num_classes (model head width)
NUM_CLASSES = 10        # get_flexible_categorical_config: num_classes (fixed; == max_num_classes)

# Swap to {'distribution': 'uniform_int', 'min': 2, 'max': NUM_CLASSES} to reproduce
# nodepfn's commented-out per-dataset class sampling instead of the active fixed value.
N_CLASSES_DIST = {'distribution': 'constant', 'value': NUM_CLASSES}

# --- Distributions copied verbatim from nodepfn (model_configs.get_diff_causal /
#     flexible_categorical / pretrain.py). Field names are the casual_graph_generation ones. ---
PRIOR_DISTRIBUTIONS = {
    'n_layers':   {'distribution': 'meta_gamma', 'max_alpha': 2, 'max_scale': 3,   'lower_bound': 2, 'round': True},  # num_layers
    'hidden':     {'distribution': 'meta_gamma', 'max_alpha': 3, 'max_scale': 100, 'lower_bound': 4, 'round': True},  # prior_mlp_hidden_dim
    'n_geo':      {'distribution': 'meta_gamma', 'max_alpha': 3, 'max_scale': 7,   'lower_bound': 2, 'round': True},  # num_causes
    'drop_rate':  {'distribution': 'meta_beta',  'scale': 0.6, 'min': 0.1, 'max': 5.0},                               # prior_mlp_dropout_prob
    'n_features': {'distribution': 'uniform_int', 'min': 1, 'max': MAX_FEATURES},                                     # num_features_used (max_features)
    'n_classes':  N_CLASSES_DIST,                                                                                     # num_classes (fixed at max_num_classes)
}


# ---------------------------------------------------------------------------
# nodepfn meta-distribution samplers (see priors/differentiable_prior.py and
# priors/utils.py: gamma_sampler_f / beta_sampler_f / uniform_int_sampler_f).
# ---------------------------------------------------------------------------
def _sample_meta_gamma(rng, max_alpha, max_scale, lower_bound, round):
    """lower_bound + [round] Gamma(shape=e^a, scale=scale/e^a), a~U(0,ln max_alpha), scale~U(0,max_scale)."""
    alpha = rng.uniform(0.0, math.log(max_alpha))
    scale = rng.uniform(0.0, max_scale)
    shape = math.exp(alpha)
    value = float(rng.gamma(shape, scale / shape))
    if round:
        return lower_bound + int(np.round(value))
    return lower_bound + value


def _sample_meta_beta(rng, scale, lo, hi):
    """scale * Beta(b, k) with b, k ~ U(lo, hi)."""
    b = rng.uniform(lo, hi)
    k = rng.uniform(lo, hi)
    return float(scale * rng.beta(b, k))


def _sample_uniform_int(rng, lo, hi):
    """round(U(lo, hi)) — matches nodepfn uniform_int_sampler_f."""
    return int(round(float(rng.uniform(lo, hi))))


def _sample_prior(rng, spec):
    dist = spec['distribution']
    if dist == 'meta_gamma':
        return _sample_meta_gamma(rng, spec['max_alpha'], spec['max_scale'],
                                  spec['lower_bound'], spec['round'])
    if dist == 'meta_beta':
        return _sample_meta_beta(rng, spec['scale'], spec['min'], spec['max'])
    if dist == 'uniform_int':
        return _sample_uniform_int(rng, spec['min'], spec['max'])
    if dist == 'constant':
        return spec['value']
    raise ValueError(f'unsupported prior distribution: {dist!r}')


def _stepped_int(rng, lo: int, hi: int, step: int) -> int:
    """Uniform draw from {lo, lo+step, ..., hi} inclusive, matching an IntSlider."""
    n = (hi - lo) // step
    return int(lo + step * int(rng.integers(0, n + 1)))


def sample_config(rng=None, seed: int | None = None, max_tries: int = 10_000, **fixed) -> GraphConfig:
    """Draw one :class:`GraphConfig` from the nodepfn prior distributions.

    SCM/dataset fields follow ``PRIOR_DISTRIBUTIONS`` (the nodepfn causal prior); the
    similarity-kernel fields follow the cell [23] dashboard ranges. Re-draws until the
    SCM constraint ``n_features + n_geo <= n_layers*hidden - 1`` holds.

    Parameters
    ----------
    rng : numpy Generator, optional
        Hyperparameter stream; created from `seed` if not given.
    seed : int, optional
        Seed for a fresh ``np.random.default_rng`` when `rng` is None.
    max_tries : int
        Cap on re-draws before raising (the constraint is almost always met quickly).
    **fixed :
        Pin any field to a constant (e.g. ``similarity='cosine'``); overrides the draw.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    for _ in range(max_tries):
        cfg = GraphConfig(
            # --- nodepfn causal-prior distributions ---
            n_layers      = _sample_prior(rng, PRIOR_DISTRIBUTIONS['n_layers']),
            hidden        = _sample_prior(rng, PRIOR_DISTRIBUTIONS['hidden']),
            n_geo         = _sample_prior(rng, PRIOR_DISTRIBUTIONS['n_geo']),
            drop_rate     = _sample_prior(rng, PRIOR_DISTRIBUTIONS['drop_rate']),
            n_features    = _sample_prior(rng, PRIOR_DISTRIBUTIONS['n_features']),
            n_classes     = _sample_prior(rng, PRIOR_DISTRIBUTIONS['n_classes']),
            # --- geometric-similarity-only fields (cell [23] dashboard ranges) ---
            n_nodes       = _stepped_int(rng, *SLIDER_RANGES['n_nodes']),
            similarity    = str(rng.choice(SIMILARITY_CHOICES)),
            sim_threshold = float(rng.uniform(-1.0, 1.0)),
            normalize     = str(rng.choice(FRAME_CHOICES)),
            sim_out_dim   = _stepped_int(rng, *SLIDER_RANGES['sim_out_dim']),
            sim_act       = str(rng.choice(SIM_ACT_CHOICES)),
        )
        for key, value in fixed.items():
            if not hasattr(cfg, key):
                raise TypeError(f'unknown GraphConfig field: {key!r}')
            setattr(cfg, key, value)
        if cfg.is_valid():
            return cfg

    raise RuntimeError(
        f'could not draw a valid GraphConfig in {max_tries} tries; the SCM constraint '
        f'n_features + n_geo <= n_layers*hidden - 1 may be unsatisfiable under the '
        f'current `fixed` overrides.'
    )
