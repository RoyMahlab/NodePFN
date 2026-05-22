import isolated_prior.priors as priors
import torch
import numpy as np

uniform_int_sampler_f = lambda a, b: lambda: round(np.random.uniform(a, b))


def make_get_batch(model_proto, **extra_kwargs):
    def new_get_batch(
        batch_size,
        seq_len,
        num_features,
        hyperparameters,
        device,
        model_proto=model_proto,
        **kwargs
    ):
        kwargs = {**extra_kwargs, **kwargs}  # new args overwrite pre-specified args
        return model_proto.get_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            hyperparameters=hyperparameters,
            num_features=num_features,
            **kwargs
        )

    return new_get_batch


config = {
    "num_features": 100,
    "hyperparameters": {
        "lr": 0.00013378709137579946,
        "dropout": 0.0,
        "emsize": 512,
        "batch_size": 1,
        "nlayers": 12,
        "num_features": 100,
        "nhead": 4,
        "nhid_factor": 2,
        "bptt": 1024,
        "eval_positions": None,
        "seq_len_used": 50,
        "sampling": "mixed",
        "epochs": 30,
        "num_steps": 8192,
        "verbose": False,
        "mix_activations": True,
        "pre_sample_causes": True,
        "multiclass_type": "rank",
        "nan_prob_unknown_reason_reason_prior": 0.5,
        "categorical_feature_p": 0.2,
        "nan_prob_no_reason": 0.0,
        "nan_prob_unknown_reason": 0.0,
        "nan_prob_a_reason": 0.0,
        "max_num_classes": 20,
        "num_classes": uniform_int_sampler_f(2, 20),
        "noise_type": "Gaussian",
        "balanced": False,
        "normalize_to_ranking": False,
        "set_value_to_nan": 0.1,
        "normalize_by_used_features": True,
        "num_features_used": 100,
        "num_categorical_features_sampler_a": -1.0,
        "differentiable_hyperparameters": {
            "distribution": "uniform",
            "min": 1000000.0,
            "max": 1000001.0,
        },
        "graph_type": "sbm",
        "homophily_rate": 0.25212484456586826,
        "p_in": 0.07797399038409931,
        "edge_prob": 0.02889311816745546,
        "prior_type": "prior_bag",
        "differentiable": True,
        "flexible": True,
        "recompute_attn": True,
        "max_features": 100,
        "is_baseline": True,
        "bptt_extra_samples": None,
        "output_multiclass_ordered_p": 0.0,
        "multiclass_loss_type": "nono",
        "new_mlp_per_example": True,
        "prior_mlp_scale_weights_sqrt": True,
        "batch_size_per_gp_sample": None,
        "normalize_ignore_label_too": True,
        "differentiable_hps_as_style": False,
        "max_eval_pos": 1000,
        "random_feature_rotation": True,
        "rotate_normalized_labels": True,
        "canonical_y_encoder": False,
        "aggregate_k_gradients": 8,
        "total_available_time_in_s": None,
        "train_mixed_precision": True,
        "efficient_eval_masking": True,
        "pos_encoder": "none",
        "prompt_dim": 4096,
        "conv_type": "gcn",
        "use_gps_style": False,
        "prior_bag_get_batch": [
            make_get_batch(
                priors.flexible_categorical,
                **{"get_batch": make_get_batch(priors.fast_gp)}
            ),
            make_get_batch(
                priors.flexible_categorical, **{"get_batch": make_get_batch(priors.mlp)}
            ),
        ],
        "prior_bag_exp_weights_1": 2.0,
        "normalize_labels": True,
        "check_is_compatible": True,
    },
    "batch_size_per_gp_sample": None,
    "prompt_dim": 4096,
    "get_batch": make_get_batch(priors.prior_bag, **{}),
    "differentiable_hyperparameters": {
        "prior_bag_exp_weights_1": {
            "distribution": "uniform",
            "min": 1000000.0,
            "max": 1000001.0,
        },
        "num_layers": {
            "distribution": "meta_gamma",
            "max_alpha": 2,
            "max_scale": 3,
            "round": True,
            "lower_bound": 2,
        },
        "prior_mlp_hidden_dim": {
            "distribution": "meta_gamma",
            "max_alpha": 3,
            "max_scale": 100,
            "round": True,
            "lower_bound": 4,
        },
        "prior_mlp_dropout_prob": {
            "distribution": "meta_beta",
            "scale": 0.6,
            "min": 0.1,
            "max": 5.0,
        },
        "noise_std": {
            "distribution": "meta_trunc_norm_log_scaled",
            "max_mean": 0.3,
            "min_mean": 0.0001,
            "round": False,
            "lower_bound": 0.0,
        },
        "init_std": {
            "distribution": "meta_trunc_norm_log_scaled",
            "max_mean": 10.0,
            "min_mean": 0.01,
            "round": False,
            "lower_bound": 0.0,
        },
        "num_causes": {
            "distribution": "meta_gamma",
            "max_alpha": 3,
            "max_scale": 7,
            "round": True,
            "lower_bound": 2,
        },
        "is_causal": {"distribution": "meta_choice", "choice_values": [True, False]},
        "pre_sample_weights": {
            "distribution": "meta_choice",
            "choice_values": [True, False],
        },
        "y_is_effect": {"distribution": "meta_choice", "choice_values": [True, False]},
        "prior_mlp_activations": {
            "distribution": "meta_choice_mixed",
            "choice_values": [torch.nn.Tanh, torch.nn.Identity, torch.nn.ReLU],
        },
        "block_wise_dropout": {
            "distribution": "meta_choice",
            "choice_values": [True, False],
        },
        "sort_features": {
            "distribution": "meta_choice",
            "choice_values": [True, False],
        },
        "in_clique": {"distribution": "meta_choice", "choice_values": [True, False]},
        "outputscale": {
            "distribution": "meta_trunc_norm_log_scaled",
            "max_mean": 10.0,
            "min_mean": 1e-05,
            "round": False,
            "lower_bound": 0,
        },
        "lengthscale": {
            "distribution": "meta_trunc_norm_log_scaled",
            "max_mean": 10.0,
            "min_mean": 1e-05,
            "round": False,
            "lower_bound": 0,
        },
        "noise": {
            "distribution": "meta_choice",
            "choice_values": [1e-05, 0.0001, 0.01],
        },
        "graph_type": {
            "distribution": "meta_choice",
            "choice_values": ["sbm", "random"],
        },
        "homophily_rate": {"distribution": "uniform", "min": 0.1, "max": 0.9},
        "p_in": {"distribution": "uniform", "min": 0.01, "max": 0.1},
        "edge_prob": {"distribution": "uniform", "min": 0.01, "max": 0.05},
    },
}
