import argparse
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from nodepfn.scripts.transformer_prediction_interface import NodePFNClassifier
import time
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import SimpleConv
from nodepfn.dataset import load_dataset
from nodepfn.data_utils import load_fixed_splits, class_rand_splits, class_rand_splits_half
from sklearn.decomposition import TruncatedSVD
import os

def fix_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def update_edge_index(edge_index, train_idx, test_idx, num_nodes):
    """Remap edge_index (global node ids) onto local [0..len(train_idx)) for train_idx
    and [len(train_idx)..len(train_idx)+len(test_idx)) for test_idx. Edges touching a node
    outside train_idx/test_idx are dropped rather than silently aliased to node 0, so this
    is safe to call with test_idx set to a chunk of the full query set."""
    all_indices = torch.cat([train_idx, test_idx])

    keep_node = torch.zeros(num_nodes, dtype=torch.bool)
    keep_node[all_indices] = True
    edge_mask = keep_node[edge_index[0]] & keep_node[edge_index[1]]
    edge_index = edge_index[:, edge_mask]

    old_to_new = torch.zeros(num_nodes, dtype=torch.long)
    old_to_new[train_idx] = torch.arange(len(train_idx))
    old_to_new[test_idx] = torch.arange(len(test_idx)) + len(train_idx)

    new_edge_index = old_to_new[edge_index]

    return new_edge_index

def run_experiments(args):
    valid_accuracies, test_accuracies = [], []
    valid_roc_aucs, test_roc_aucs = [], []
    fit_times, pred_times = [], []
    dataset = load_dataset(args.data_dir, args.dataset)
    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)

    if args.dataset in ['ogbn-arxiv', 'ogbn-products', 'ogbn-proteins']:
        split_idx_lst = [dataset.load_fixed_splits() for _ in range(args.runs)]
    elif args.rand_split:
        split_idx_lst = [dataset.get_idx_split(split_type='random', train_prop=args.train_prop, valid_prop=args.valid_prop)
                         for _ in range(args.runs)]
    elif args.rand_split_class:
        split_idx_lst = [class_rand_splits(
            dataset.label, args.label_num_per_class, args.valid_num, args.test_num)]
    elif args.rand_split_class_half:
        split_idx_lst = [class_rand_splits_half(
            dataset.label, args.label_num_per_class)]
    else:
        if hasattr(dataset, 'train_mask'):
           # print("Using built-in data splits.")
            split_idx_lst = []
            for split_num in range(args.runs):
                split_idx = {}
                train_mask = dataset.train_mask if dataset.train_mask.dim() == 1 else dataset.train_mask[:, split_num]
                val_mask = dataset.val_mask if dataset.val_mask.dim() == 1 else dataset.val_mask[:, split_num]
                test_mask = dataset.test_mask if dataset.test_mask.dim() == 1 else dataset.test_mask[:, split_num]

                split_idx['train'] = train_mask.nonzero(as_tuple=False).view(-1)
                split_idx['valid'] = val_mask.nonzero(as_tuple=False).view(-1)
                split_idx['test'] = test_mask.nonzero(as_tuple=False).view(-1)
                split_idx_lst.append(split_idx)
        else :
           # print("Using standard Planetoid splits (20/class, 500 val, 1000 test).")
            split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset, label_num_per_class=args.label_num_per_class)

    
    valid_accuracies = []
    test_accuracies = []
    valid_roc_aucs = []
    test_roc_aucs = []
    test_ap = []
    fit_times = []

    for run in range(args.runs):
        print(f"\n================ Run {run+1}/{args.runs} ================")
        args.seed = run
        fix_seed(args.seed)
        split_idx = split_idx_lst[run] if run < len(split_idx_lst) else split_idx_lst[0]
        train_idx = split_idx['train']
        valid_idx = split_idx['valid']
        test_idx = split_idx['test']

        n = dataset.graph['num_nodes']
        # Process edges
        dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
        dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
        dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)

        # Add edge_weight using gcn_norm
        dataset.graph['edge_index'], edge_weight = gcn_norm(
            dataset.graph['edge_index'],
            edge_weight=None,
            num_nodes=n,
            add_self_loops=True
        )        
        dataset.graph['edge_weight'] = edge_weight

        conv = SimpleConv(aggr='sum')
        
        # Convert to proper device and apply smoothing
        X = dataset.graph['node_feat']
        edge_index = dataset.graph['edge_index']
        y = dataset.label.squeeze().numpy()
        # Apply multiple smoothing steps
        for step in range(args.smoothing_steps):
            X = conv(X, dataset.graph['edge_index'], dataset.graph['edge_weight'])

        original_features = X.shape[1]

        if args.dim_reduction != 'none':
            n_components = min(args.n_components, original_features, X.shape[0] - 1)
            if args.dim_reduction == 'tsvd':
                reducer = TruncatedSVD(n_components=n_components, algorithm=args.svd_algorithm, random_state=args.seed)
                X = reducer.fit_transform(X)
        else:
            print(f"No dimensionality reduction applied. Using all {original_features} features.")

        all_idx = torch.arange(X.shape[0])
        train_idx_set = set(train_idx.tolist())
        query_idx = torch.tensor([i for i in all_idx.tolist() if i not in train_idx_set], dtype=torch.long)

        if args.train_sample_size is not None and len(train_idx) > args.train_sample_size:
            # Train tokens self-attend to each other (O(train_size^2)), so datasets with a large
            # fixed train split (e.g. ogbn-arxiv's ~91k-node OGB split) need the in-context example
            # set capped regardless of GPU memory; leftover train nodes are dropped, not added to query.
            rng = np.random.RandomState(args.seed)
            sample = np.sort(rng.choice(len(train_idx), size=args.train_sample_size, replace=False))
            print(f"Subsampling train set from {len(train_idx)} to {args.train_sample_size} in-context examples")
            train_idx = train_idx[torch.from_numpy(sample)]

        X_train = X[train_idx]
        y_train = y[train_idx]
        X_query = X[query_idx]
        y_query = y[query_idx]

        valid_mask = torch.isin(query_idx, valid_idx)
        test_mask = torch.isin(query_idx, test_idx)

        edge_index_run = update_edge_index(edge_index, train_idx, query_idx, num_nodes=n)
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Query set (valid + test): {X_query.shape[0]} samples")
        print(f"  - Valid: {len(valid_idx)} samples")
        print(f"  - Test: {len(test_idx)} samples")

        start_time = time.time()
        base_model_path = args.base_model_path
        clf = NodePFNClassifier(device=args.compute_device, base_path=base_model_path,
                               N_ensemble_configurations=args.n_ensemble,
                               seed=args.seed,
                               batch_size_inference=args.batch_size_inference,
                               subsample_features=True,
                               i=0, e=args.e,
                               fp16_inference=args.fp16_inference,
                               amp_dtype=args.amp_dtype,
                               pipeline_devices=args.pipeline_devices)

        clf.fit(X_train, y_train, edge_index_run, overwrite_warning=True)
        fit_time = time.time() - start_time

        if args.query_batch_size is not None and X_query.shape[0] > args.query_batch_size:
            prob_chunks = []
            for start in range(0, X_query.shape[0], args.query_batch_size):
                chunk_query_idx = query_idx[start:start + args.query_batch_size]
                # local_model (GCN) needs edge_index consistent with whichever nodes are
                # in this call's X_full (train + this chunk); edges to query nodes outside
                # the chunk are dropped, so cross-chunk query-query smoothing is lost.
                clf.edge_index = update_edge_index(edge_index, train_idx, chunk_query_idx, num_nodes=n)
                prob_chunks.append(
                    clf.predict_proba(X_query[start:start + args.query_batch_size], normalize_with_test=True)
                )
            prediction_probabilities = np.concatenate(prob_chunks, axis=0)
        else:
            prediction_probabilities = clf.predict_proba(X_query, normalize_with_test=True)
        predictions = clf.classes_.take(np.argmax(prediction_probabilities, axis=-1))
        p_eval = prediction_probabilities.max(axis=-1)

        y_valid = y_query[valid_mask]
        y_test = y_query[test_mask]
        pred_valid = predictions[valid_mask]
        pred_test = predictions[test_mask]
        prob_valid = prediction_probabilities[valid_mask]
        prob_test = prediction_probabilities[test_mask]

        accuracy_valid = accuracy_score(y_valid, pred_valid)
        accuracy_test = accuracy_score(y_test, pred_test)

        roc_auc_valid = None
        roc_auc_test = None
        # Count the LABELLED classes only. load_graphland marks unlabelled nodes with -1
        # (dataset.py:302-307); those nodes are outside every split mask, so counting them
        # here would route a genuinely binary dataset (city-reviews: {-1, 0, 1}) into the
        # multiclass branch, where ovr on 2-column probabilities raises and both ROC-AUC and
        # average precision are silently lost.
        if len(np.unique(y[y >= 0])) == 2:
            test_ap.append(average_precision_score(y_test, prob_test[:, 1]))
            roc_auc_valid = roc_auc_score(y_valid, prob_valid[:, 1])
            roc_auc_test = roc_auc_score(y_test, prob_test[:, 1])
        else:
            try:
                roc_auc_valid = roc_auc_score(y_valid, prob_valid, multi_class='ovr')
                roc_auc_test = roc_auc_score(y_test, prob_test, multi_class='ovr')
            except Exception as e:  # keep the sweep going, but say why the metric is missing
                print(f"WARNING: multiclass ROC-AUC unavailable ({type(e).__name__}: {e})")
                roc_auc_valid = None
                roc_auc_test = None

        valid_accuracies.append(accuracy_valid)
        test_accuracies.append(accuracy_test)
        valid_roc_aucs.append(roc_auc_valid)
        test_roc_aucs.append(roc_auc_test)
        fit_times.append(fit_time)

        print(f"Run {run+1}: val acc={accuracy_valid:.4f}, test acc={accuracy_test:.4f}")

    print("\n================ Summary ================")
    print(f"Validation Accuracy: {np.mean(valid_accuracies)*100:.2f} ± {np.std(valid_accuracies)*100:.2f}")
    print(f"Test Accuracy: {np.mean(test_accuracies)*100:.2f} ± {np.std(test_accuracies)*100:.2f}")
    if test_ap:
        print(f"Test Average Precision: {np.mean(test_ap)*100:.2f} ± {np.std(test_ap)*100:.2f}")
    have_valid_roc = all(v is not None for v in valid_roc_aucs)
    have_test_roc = all(v is not None for v in test_roc_aucs)
    if have_valid_roc:
        print(f"Validation ROC AUC: {np.mean(valid_roc_aucs)*100:.2f} ± {np.std(valid_roc_aucs)*100:.2f}")
    if have_test_roc:
        print(f"Test ROC AUC: {np.mean(test_roc_aucs)*100:.2f} ± {np.std(test_roc_aucs)*100:.2f}")
    print(f"Fit time (mean ± std): {np.mean(fit_times):.2f} ± {np.std(fit_times):.2f} sec")

    results = {
        'dataset': args.dataset,
        'runs': args.runs,
        'valid_acc_mean': float(np.mean(valid_accuracies)),
        'valid_acc_std': float(np.std(valid_accuracies)),
        'test_acc_mean': float(np.mean(test_accuracies)),
        'test_acc_std': float(np.std(test_accuracies)),
        'valid_rocauc_mean': float(np.mean(valid_roc_aucs)) if have_valid_roc else None,
        'valid_rocauc_std': float(np.std(valid_roc_aucs)) if have_valid_roc else None,
        'test_rocauc_mean': float(np.mean(test_roc_aucs)) if have_test_roc else None,
        'test_rocauc_std': float(np.std(test_roc_aucs)) if have_test_roc else None,
        # average_precision_score is only computed for binary datasets, so test_ap is empty
        # for multiclass ones; np.mean([]) would be a NaN plus three RuntimeWarnings.
        'test_ap_mean': float(np.mean(test_ap)) if test_ap else None,
        'test_ap_std': float(np.std(test_ap)) if test_ap else None,
        'fit_time_mean': float(np.mean(fit_times)),
        'fit_time_std': float(np.std(fit_times)),
        'config': {
            'dim_reduction': args.dim_reduction,
            'n_components': args.n_components,
            'svd_algorithm': args.svd_algorithm,
            'smoothing_steps': args.smoothing_steps,
            'n_ensemble': args.n_ensemble,
            'label_num_per_class': args.label_num_per_class,
            'base_model_path': args.base_model_path,
            'cpu': args.cpu,
        },
    }
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NodePFN Test for Node Classification')
    parser.add_argument('--dataset', type=str, default='roman-empire')
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use for single-GPU inference (default: 0)')
    parser.add_argument('--pipeline_gpus', type=int, default=1,
                        help='split the transformer layers across this many GPUs (cuda:0..cuda:N-1) '
                             'to cap per-GPU memory on large graphs (default: 1, single GPU)')
    parser.add_argument('--precision', type=str, default='fp32',
                        choices=['fp32', 'fp16', 'bf16'],
                        help='inference precision; fp16/bf16 roughly halve memory (default: fp32)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=1,
                        help='number of distinct runs')
    parser.add_argument('--train_prop', type=float, default=.5,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.25,
                        help='validation label proportion')
    parser.add_argument('--rand_split', action='store_true',
                        help='use random splits')
    parser.add_argument('--rand_split_class', action='store_true',
                        help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--rand_split_class_half', action='store_true',
                        help='use random splits with a fixed number of labeled nodes for each class')
    
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=1000,
                        help='Total number of test')
    
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc'],
                        help='evaluation metric')
    
    parser.add_argument('--batch_size_inference', type=int, default=32)
    parser.add_argument('--query_batch_size', type=int, default=None,
                        help='split prediction over the query set into chunks of this many rows, '
                             'to cap the (train+query) attention context on large graphs (default: no chunking)')
    parser.add_argument('--train_sample_size', type=int, default=None,
                        help='cap the in-context train set to this many nodes (train tokens self-attend '
                             'to each other, so this is independent of --query_batch_size; default: no cap)')
    parser.add_argument('--base_model_path', type=str, default='models_ckpts/pfn/')
    parser.add_argument('--e', type=int, default=30)
    parser.add_argument('--dim_reduction', type=str, default='none',
                        choices=['none', 'tsvd'], help='Dimensionality reduction method to apply to node features')
    parser.add_argument('--n_components', type=int, default=50)
    parser.add_argument('--svd_algorithm', type=str, default='arpack',
                        choices=['arpack', 'randomized'])
    parser.add_argument('--smoothing_steps', type=int, default=0, 
                       help='Number of smoothing steps for feature smoothing')
    parser.add_argument('--n_ensemble', type=int, default=32,
                       help='Number of ensemble configurations for NodePFN')
    parser.add_argument('--results_json', type=str, default=None,
                       help='If set, write the summary results dict as JSON to this path')
    args = parser.parse_args()

    # Resolve precision -> (autocast enabled, dtype).
    _precision_map = {
        'fp32': (False, None),
        'fp16': (True, torch.float16),
        'bf16': (True, torch.bfloat16),
    }
    args.fp16_inference, args.amp_dtype = _precision_map[args.precision]

    # Resolve compute device(s). Pipelining spreads the layers across
    # cuda:0..cuda:(pipeline_gpus-1); otherwise run on a single device.
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

    print(f"Testing NodePFN on {args.dataset} dataset "
          f"(device={args.compute_device}, pipeline_gpus={args.pipeline_gpus}, precision={args.precision})")

    results = run_experiments(args)

    if args.results_json is not None:
        import json
        with open(args.results_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Wrote results to {args.results_json}")