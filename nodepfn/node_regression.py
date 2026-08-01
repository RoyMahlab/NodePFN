import argparse
import time
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import TruncatedSVD
from scipy.stats import spearmanr
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import SimpleConv

from nodepfn.scripts.transformer_prediction_interface import NodePFNClassifier
from nodepfn.dataset import load_bluesky_splits
from nodepfn.node_classification import update_edge_index, fix_seed

TARGET_NAMES = ['likes', 'replies', 'reposts']


def preprocess_edges(edge_index, num_nodes):
    edge_index = to_undirected(edge_index)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    edge_index, edge_weight = gcn_norm(edge_index, edge_weight=None, num_nodes=num_nodes, add_self_loops=True)
    return edge_index, edge_weight


def smooth_features(X, edge_index, edge_weight, steps):
    conv = SimpleConv(aggr='sum')
    for _ in range(steps):
        X = conv(X, edge_index, edge_weight)
    return X


def bucketize_train(y_train, n_bins):
    """Quantile-bin a continuous target using TRAIN data only. Returns bin edges
    and, per bin, the mean training y (used to decode a predicted class
    distribution back into a continuous value)."""
    edges = np.unique(np.quantile(y_train, np.linspace(0, 1, n_bins + 1)))
    n_bins_eff = max(len(edges) - 1, 1)
    bin_idx = np.clip(np.digitize(y_train, edges[1:-1], right=False), 0, n_bins_eff - 1)
    bin_values = np.array([
        y_train[bin_idx == b].mean() if np.any(bin_idx == b) else 0.5 * (edges[b] + edges[b + 1])
        for b in range(n_bins_eff)
    ])
    return edges, bin_values, bin_idx


def subsample_train_context(X_train, y_train_binned, edge_index_train, n_train, sample_size, seed):
    """NodePFN's train tokens self-attend to each other (O(train_size^2)), so the
    in-context example set must stay small regardless of GPU memory -- unlike the
    query set, this can't be fixed by chunking. Sample a small subgraph of the train
    graph to serve as the model's context."""
    if sample_size is None or n_train <= sample_size:
        return X_train, y_train_binned, edge_index_train, n_train
    rng = np.random.RandomState(seed)
    sample_idx = np.sort(rng.choice(n_train, size=sample_size, replace=False))
    sample_idx_t = torch.from_numpy(sample_idx).long()
    empty = torch.empty(0, dtype=torch.long)
    edge_index_sampled = update_edge_index(edge_index_train, sample_idx_t, empty, num_nodes=n_train)
    return X_train[sample_idx], y_train_binned[sample_idx], edge_index_sampled, sample_size


def predict_regression(clf, X_train_local_size, X_query, train_idx, edge_index, num_nodes,
                        bin_values, query_batch_size):
    """Predict continuous values for X_query by discretized-classification + expectation
    decoding. Chunks the query set (with a matching subgraph edge_index per chunk, since
    the model's local GCN branch needs edge_index consistent with whichever nodes are in
    the current forward pass) to bound GPU memory on large graphs."""
    n_query = X_query.shape[0]
    query_idx_all = torch.arange(X_train_local_size, X_train_local_size + n_query)
    batch = query_batch_size or n_query

    prob_chunks = []
    for start in range(0, n_query, batch):
        chunk_query_idx = query_idx_all[start:start + batch]
        clf.edge_index = update_edge_index(edge_index, train_idx, chunk_query_idx, num_nodes=num_nodes)
        prob_chunks.append(clf.predict_proba(X_query[start:start + batch], normalize_with_test=True))
    probs = np.concatenate(prob_chunks, axis=0)

    bin_values_ordered = bin_values[clf.classes_.astype(int)]
    return probs @ bin_values_ordered


def run_experiments(args):
    splits = load_bluesky_splits(args.dataset)
    train_data, val_data, test_data = splits['train'], splits['val'], splits['test']

    n_train = train_data.num_nodes
    n_val = val_data.num_nodes
    n_test = test_data.num_nodes
    print(f"Train graph: {n_train} nodes, {train_data.edge_index.shape[1]} edges")
    print(f"Val graph:   {n_val} nodes, {val_data.edge_index.shape[1]} edges")
    print(f"Test graph:  {n_test} nodes, {test_data.edge_index.shape[1]} edges")

    targets = TARGET_NAMES if args.target == 'all' else [args.target]
    all_results = {}

    for target_name in targets:
        target_idx = TARGET_NAMES.index(target_name)
        print(f"\n================ Target: {target_name} ================")

        val_metrics = {'mse': [], 'mae': [], 'r2': [], 'spearman': []}
        test_metrics = {'mse': [], 'mae': [], 'r2': [], 'spearman': []}
        fit_times = []

        for run in range(args.runs):
            args.seed = run
            fix_seed(args.seed)
            print(f"\n---- Run {run + 1}/{args.runs} ----")

            edge_index_train, ew_train = preprocess_edges(train_data.edge_index, n_train)
            edge_index_val, ew_val = preprocess_edges(val_data.edge_index, n_val)
            edge_index_test, ew_test = preprocess_edges(test_data.edge_index, n_test)

            X_train = smooth_features(train_data.x, edge_index_train, ew_train, args.smoothing_steps)
            X_val = smooth_features(val_data.x, edge_index_val, ew_val, args.smoothing_steps)
            X_test = smooth_features(test_data.x, edge_index_test, ew_test, args.smoothing_steps)

            if args.dim_reduction != 'none':
                n_components = min(args.n_components, X_train.shape[1], X_train.shape[0] - 1)
                reducer = TruncatedSVD(n_components=n_components, algorithm=args.svd_algorithm, random_state=args.seed)
                X_train = reducer.fit_transform(X_train)
                X_val = reducer.transform(X_val)
                X_test = reducer.transform(X_test)
            else:
                X_train, X_val, X_test = X_train.numpy(), X_val.numpy(), X_test.numpy()

            y_train = train_data.y.numpy()[:, target_idx]
            y_val = val_data.y.numpy()[:, target_idx]
            y_test = test_data.y.numpy()[:, target_idx]

            _, bin_values, y_train_binned = bucketize_train(y_train, args.n_bins)
            print(f"Discretized into {len(bin_values)} quantile bins (requested {args.n_bins})")

            X_train, y_train_binned, edge_index_train, n_train_ctx = subsample_train_context(
                X_train, y_train_binned, edge_index_train, n_train, args.train_sample_size, args.seed
            )
            print(f"Using {n_train_ctx} of {n_train} train nodes as in-context examples")
            train_idx = torch.arange(n_train_ctx)

            start_time = time.time()
            clf = NodePFNClassifier(device=args.compute_device, base_path=args.base_model_path,
                                     N_ensemble_configurations=args.n_ensemble,
                                     seed=args.seed,
                                     batch_size_inference=args.batch_size_inference,
                                     subsample_features=True,
                                     i=0, e=args.e,
                                     fp16_inference=args.fp16_inference,
                                     amp_dtype=args.amp_dtype,
                                     pipeline_devices=args.pipeline_devices)
            clf.fit(X_train, y_train_binned, edge_index_train, overwrite_warning=True)
            fit_time = time.time() - start_time
            fit_times.append(fit_time)

            for split_name, X_query, y_query_cont, edge_index_query, num_query, metrics in (
                ('val', X_val, y_val, edge_index_val, n_val, val_metrics),
                ('test', X_test, y_test, edge_index_test, n_test, test_metrics),
            ):
                combined_edge_index = torch.cat(
                    [edge_index_train, edge_index_query + n_train_ctx], dim=1
                )
                y_pred = predict_regression(
                    clf, n_train_ctx, X_query, train_idx, combined_edge_index,
                    num_nodes=n_train_ctx + num_query, bin_values=bin_values,
                    query_batch_size=args.query_batch_size,
                )

                mse = mean_squared_error(y_query_cont, y_pred)
                mae = mean_absolute_error(y_query_cont, y_pred)
                r2 = r2_score(y_query_cont, y_pred)
                rho = spearmanr(y_query_cont, y_pred).correlation

                metrics['mse'].append(mse)
                metrics['mae'].append(mae)
                metrics['r2'].append(r2)
                metrics['spearman'].append(rho)
                print(f"  {split_name}: mse={mse:.4f} mae={mae:.4f} r2={r2:.4f} spearman={rho:.4f}")

        print(f"\n---- {target_name} summary over {args.runs} run(s) ----")
        for split_name, metrics in (('Validation', val_metrics), ('Test', test_metrics)):
            print(f"{split_name} MSE:      {np.mean(metrics['mse']):.4f} ± {np.std(metrics['mse']):.4f}")
            print(f"{split_name} MAE:      {np.mean(metrics['mae']):.4f} ± {np.std(metrics['mae']):.4f}")
            print(f"{split_name} R2:       {np.mean(metrics['r2']):.4f} ± {np.std(metrics['r2']):.4f}")
            print(f"{split_name} Spearman: {np.mean(metrics['spearman']):.4f} ± {np.std(metrics['spearman']):.4f}")
        print(f"Fit time (mean ± std): {np.mean(fit_times):.2f} ± {np.std(fit_times):.2f} sec")

        all_results[target_name] = {
            'val_mse_mean': float(np.mean(val_metrics['mse'])), 'val_mse_std': float(np.std(val_metrics['mse'])),
            'val_mae_mean': float(np.mean(val_metrics['mae'])), 'val_mae_std': float(np.std(val_metrics['mae'])),
            'val_r2_mean': float(np.mean(val_metrics['r2'])), 'val_r2_std': float(np.std(val_metrics['r2'])),
            'val_spearman_mean': float(np.mean(val_metrics['spearman'])), 'val_spearman_std': float(np.std(val_metrics['spearman'])),
            'test_mse_mean': float(np.mean(test_metrics['mse'])), 'test_mse_std': float(np.std(test_metrics['mse'])),
            'test_mae_mean': float(np.mean(test_metrics['mae'])), 'test_mae_std': float(np.std(test_metrics['mae'])),
            'test_r2_mean': float(np.mean(test_metrics['r2'])), 'test_r2_std': float(np.std(test_metrics['r2'])),
            'test_spearman_mean': float(np.mean(test_metrics['spearman'])), 'test_spearman_std': float(np.std(test_metrics['spearman'])),
            'fit_time_mean': float(np.mean(fit_times)), 'fit_time_std': float(np.std(fit_times)),
        }

    return {'dataset': args.dataset, 'runs': args.runs, 'n_bins': args.n_bins, 'targets': all_results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NodePFN regression on BlueSky (via discretized classification)')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['bluesky_quotes', 'bluesky_replies', 'bluesky_reposts'])
    parser.add_argument('--target', type=str, default='all', choices=['likes', 'replies', 'reposts', 'all'])
    parser.add_argument('--n_bins', type=int, default=100,
                        help='number of quantile bins used to discretize the continuous target '
                             "(must be <= the checkpoint's max_num_classes)")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--pipeline_gpus', type=int, default=1,
                        help='split the transformer layers across this many GPUs (default: 1, single GPU)')
    parser.add_argument('--precision', type=str, default='fp32', choices=['fp32', 'fp16', 'bf16'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--batch_size_inference', type=int, default=32)
    parser.add_argument('--query_batch_size', type=int, default=None,
                        help='split prediction over the query graph into chunks of this many rows, '
                             'to cap the (train+query) attention/GCN context on large graphs')
    parser.add_argument('--base_model_path', type=str, default='models_ckpts/geo_100_classes')
    parser.add_argument('--e', type=int, default=50)
    parser.add_argument('--dim_reduction', type=str, default='none', choices=['none', 'tsvd'])
    parser.add_argument('--n_components', type=int, default=50)
    parser.add_argument('--svd_algorithm', type=str, default='arpack', choices=['arpack', 'randomized'])
    parser.add_argument('--smoothing_steps', type=int, default=0)
    parser.add_argument('--n_ensemble', type=int, default=32)
    parser.add_argument('--train_sample_size', type=int, default=1000,
                         help='train tokens self-attend to each other (O(n^2)), so the in-context '
                              'example set is subsampled from the train graph to this many nodes')
    parser.add_argument('--results_json', type=str, default=None)
    args = parser.parse_args()

    _precision_map = {'fp32': (False, None), 'fp16': (True, torch.float16), 'bf16': (True, torch.bfloat16)}
    args.fp16_inference, args.amp_dtype = _precision_map[args.precision]

    if args.cpu:
        args.compute_device = 'cpu'
        args.pipeline_devices = None
    elif args.pipeline_gpus > 1:
        n_avail = torch.cuda.device_count()
        if args.pipeline_gpus > n_avail:
            raise SystemExit(f"--pipeline_gpus={args.pipeline_gpus} but only {n_avail} CUDA device(s) visible")
        args.pipeline_devices = [f'cuda:{i}' for i in range(args.pipeline_gpus)]
        args.compute_device = args.pipeline_devices[0]
    else:
        args.compute_device = f'cuda:{args.device}'
        args.pipeline_devices = None

    print(f"Testing NodePFN regression on {args.dataset} "
          f"(device={args.compute_device}, pipeline_gpus={args.pipeline_gpus}, precision={args.precision})")

    results = run_experiments(args)

    if args.results_json is not None:
        import json
        with open(args.results_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Wrote results to {args.results_json}")
