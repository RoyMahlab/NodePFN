import argparse
import pandas as pd
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
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

def update_edge_index(edge_index, train_idx, test_idx):
    all_indices = torch.cat([train_idx, test_idx])
    
    old_to_new = torch.zeros(all_indices.max() + 1, dtype=torch.long)
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

        X_train = X[train_idx]
        y_train = y[train_idx]
        X_query = X[query_idx]
        y_query = y[query_idx]

        valid_mask = torch.isin(query_idx, valid_idx)
        test_mask = torch.isin(query_idx, test_idx)

        edge_index_run = update_edge_index(edge_index, train_idx, query_idx)
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Query set (valid + test): {X_query.shape[0]} samples")
        print(f"  - Valid: {len(valid_idx)} samples")
        print(f"  - Test: {len(test_idx)} samples")

        start_time = time.time()
        base_model_path = args.base_model_path
        prompt_embeddings = None
        if args.prompts_file is not None:
            from nodepfn.ginat.embed_text import load_or_embed_prompts
            prompt_embeddings = load_or_embed_prompts(
                prompts_file=args.prompts_file,
                hf_model=args.hf_model,
                device=torch.device('cpu' if args.cpu else 'cuda'),
                cache_dir=args.prompt_cache_dir,
            )
        clf = NodePFNClassifier(device='cpu' if args.cpu else 'cuda', base_path=base_model_path,
                               N_ensemble_configurations=args.n_ensemble,
                               seed=args.seed,
                               batch_size_inference=args.batch_size_inference,
                               subsample_features=True,
                               prompt_embeddings=prompt_embeddings,
                               prompt_dim=args.prompt_dim,
                               i=0, e=args.e)
        
        clf.fit(X_train, y_train, edge_index_run, overwrite_warning=True)
        fit_time = time.time() - start_time

        predictions, p_eval = clf.predict(X_query, normalize_with_test=True, return_winning_probability=True)
        prediction_probabilities = clf.predict_proba(X_query, normalize_with_test=True)
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
        if len(np.unique(y)) == 2:
            roc_auc_valid = roc_auc_score(y_valid, prob_valid[:, 1])
            roc_auc_test = roc_auc_score(y_test, prob_test[:, 1])
        else:
            try:
                roc_auc_valid = roc_auc_score(y_valid, prob_valid, multi_class='ovr')
                roc_auc_test = roc_auc_score(y_test, prob_test, multi_class='ovr')
            except:
                roc_auc_valid = None
                roc_auc_test = None

        valid_accuracies.append(accuracy_valid)
        test_accuracies.append(accuracy_test)
        valid_roc_aucs.append(roc_auc_valid)
        test_roc_aucs.append(roc_auc_test)
        fit_times.append(fit_time)

        print(f"Run {run+1}: val acc={accuracy_valid:.4f}, test acc={accuracy_test:.4f}")

    print("\n================ Summary ================")
    fit_times = np.array(fit_times)
    test_acc, test_std = np.mean(test_accuracies)*100, np.std(test_accuracies)*100
    val_acc, val_std = np.mean(valid_accuracies)*100, np.std(valid_accuracies)*100
    fit_times_mean, fit_times_std = np.mean(fit_times), np.std(fit_times)
    print(f"Test Accuracy: {test_acc:.2f} ± {test_std:.2f}")
    print(f"Validation Accuracy: {val_acc:.2f} ± {val_std:.2f}")
    exp_name = args.base_model_path.split('/')[-1]
    os.makedirs(f'results/{exp_name}', exist_ok=True)
    pd.DataFrame({
        'valid_accuracy_mean': val_acc,
        'valid_accuracy_std': val_std,
        'test_accuracy_mean': test_acc,
        'test_accuracy_std': test_std,
        'fit_time_sec_mean': fit_times_mean,
        'fit_time_sec_std': fit_times_std
    }, index=[0]).to_csv(f'results/{exp_name}/{args.dataset}_nodepfn_results_{exp_name}.csv', index=False)
    if all(v is not None for v in valid_roc_aucs):
        print(f"Validation ROC AUC: {np.mean(valid_roc_aucs)*100:.2f} ± {np.std(valid_roc_aucs)*100:.2f}")
    if all(v is not None for v in test_roc_aucs):
        print(f"Test ROC AUC: {np.mean(test_roc_aucs)*100:.2f} ± {np.std(test_roc_aucs)*100:.2f}")
    print(f"Fit time (mean ± std): {np.mean(fit_times):.2f} ± {np.std(fit_times):.2f} sec")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NodePFN Test for Node Classification')
    parser.add_argument('--dataset', type=str, default='roman-empire')
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
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
    parser.add_argument('--prompts_file', type=str, default=None, help='Path to the prompts JSON file')
    parser.add_argument('--hf_model', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--prompt_cache_dir', type=str, default=None, help='Cache directory for prompt embeddings')
    parser.add_argument('--prompt_dim', type=int, default=4096)
    
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=1000,
                        help='Total number of test')
    
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc'],
                        help='evaluation metric')
    
    parser.add_argument('--batch_size_inference', type=int, default=32)
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
    args = parser.parse_args()
    print(f"Testing NodePFN on {args.dataset} dataset")

    run_experiments(args)