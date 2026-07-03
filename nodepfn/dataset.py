import os, json
import importlib.util
import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import (
    Amazon, Coauthor, WikiCS, Planetoid, Reddit,
    CoraFull, WebKB, WikipediaNetwork, HeterophilousGraphDataset, Actor, DeezerEurope, Airports
)
from ogb.nodeproppred import NodePropPredDataset, PygNodePropPredDataset

import scipy
import gdown

# -------------------- Wrapper --------------------
class NCDataset:
    def __init__(self, name):
        self.name = name
        self.graph = {}
        self.label = None

    def __getitem__(self, idx):
        assert idx == 0
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name})'


def wrap_tg_dataset(name, data):
    ds = NCDataset(name)
    ds.graph = {
        'edge_index': data.edge_index,
        'node_feat': data.x,
        'edge_feat': None,
        'num_nodes': data.num_nodes
    }
    ds.label = data.y
    if hasattr(data, 'train_mask'):
        ds.train_mask = data.train_mask
    if hasattr(data, 'val_mask'):
        ds.val_mask = data.val_mask
    if hasattr(data, 'test_mask'):
        ds.test_mask = data.test_mask
    return ds


# -------------------- Dataset Loading Functions --------------------
def load_planetoid(data_dir, name):
    d = Planetoid(root=f'{data_dir}/Planetoid', name=name,)[0]
    return wrap_tg_dataset(name, d)

def load_amazon(data_dir, name):
    sub = 'Photo' if name == 'amazon-photo' else 'Computers'
    d = Amazon(root=f'{data_dir}/Amazon', name=sub)[0]
    return wrap_tg_dataset(name, d)

def load_coauthor(data_dir, name):
    sub = 'CS' if name == 'coauthor-cs' else 'Physics'
    d = Coauthor(root=f'{data_dir}/Coauthor', name=sub, transform=T.NormalizeFeatures())[0]
    return wrap_tg_dataset(name, d)

def load_wikics(data_dir):
    d = WikiCS(root=f'{data_dir}/wikics')[0]
    return wrap_tg_dataset('wikics', d)

def load_reddit(data_dir):
    d = Reddit(root=f'{data_dir}/Reddit')[0]
    return wrap_tg_dataset('reddit', d)

def load_cora_full(data_dir):
    d = CoraFull(root=f'{data_dir}/CoraFull')[0]
    return wrap_tg_dataset('cora-full', d)

def load_webkb(data_dir, name):
    d = WebKB(root=f'{data_dir}/WebKB', name=name.capitalize())[0]
    return wrap_tg_dataset(name, d)

def load_wikipedia_network(data_dir, name):
    d = WikipediaNetwork(root=f'{data_dir}/WikipediaNetwork', name=name, geom_gcn_preprocess=True)[0]
    return wrap_tg_dataset(name, d)

def load_wikipedia_network_squirrel(data_dir, name):
    # Squirrel
    path= f'{data_dir}/geom-gcn/{name}/{name}_filtered.npz'
    data=np.load(path)
    node_feat=data['node_features'] # unnormalized
    labels=data['node_labels']
    edges=data['edges'] #(E, 2)
    edge_index=edges.T

    dataset = NCDataset(name)

    edge_index=torch.as_tensor(edge_index)
    node_feat=torch.as_tensor(node_feat)
    labels=torch.as_tensor(labels)

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': node_feat.shape[0]}
    dataset.label = labels

    return dataset

def load_hetero(data_dir, name):
    torch_dataset = HeterophilousGraphDataset(name=name.capitalize(), root=data_dir)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes

    dataset = NCDataset(name)

    ## dataset splits are implemented in data_utils.py
    '''
    dataset.train_idx = torch.where(data.train_mask[:,0])[0]
    dataset.valid_idx = torch.where(data.val_mask[:,0])[0]
    dataset.test_idx = torch.where(data.test_mask[:,0])[0]
    '''

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label

    return dataset

def load_actor(data_dir):
    d = Actor(root=f'{data_dir}/Actor', transform=T.NormalizeFeatures())[0]
    return wrap_tg_dataset('actor', d)

def load_deezer_europe(data_dir):
    d = DeezerEurope(root=f'{data_dir}/DeezerEurope')[0]
    return wrap_tg_dataset('deezer-europe', d)


# -------------------- OGB --------------------
def load_ogb(data_dir, name):
    ogb = NodePropPredDataset(name=name, root=f'{data_dir}/ogb')
    g, labels = ogb[0]
    ds = NCDataset(name)
    ds.graph = {
        'edge_index': torch.as_tensor(g['edge_index']),
        'node_feat': torch.as_tensor(g['node_feat']),
        'edge_feat': None,
        'num_nodes': g['num_nodes']
    }
    ds.label = torch.as_tensor(labels).reshape(-1, 1)
    ds.load_fixed_splits = lambda: {k: torch.as_tensor(v) for k, v in ogb.get_idx_split().items()}
    return ds

def load_ogb_proteins(data_dir, name):
    ogb = NodePropPredDataset(name=name, root=f'{data_dir}/ogb')
    pyogb = PygNodePropPredDataset(name=name, root=f'{data_dir}/ogb', transform=T.ToSparseTensor(attr='edge_attr'))
    g, labels = ogb[0]
    data_temp = pyogb[0]
    ds = NCDataset(name)
    ds.graph = {
        'edge_index': torch.as_tensor(g['edge_index']),
        'node_feat': torch.as_tensor(data_temp.adj_t.mean(dim=1)),
        'edge_feat': None,
        'num_nodes': g['num_nodes']
    }
    del data_temp
    ds.label = torch.as_tensor(labels).reshape(-1, 1)
    ds.load_fixed_splits = lambda: {k: torch.as_tensor(v) for k, v in ogb.get_idx_split().items()}
    return ds


# -------------------- Airports --------------------
def load_air(data_dir, name):
    d = Airports(root=f'{data_dir}/Airports', name=name)[0]
    return wrap_tg_dataset(f'air-{name}', d)

def load_blogcatalog(data_dir):
    bc = os.path.join(data_dir, 'blogcatalog')
    edges = pd.read_csv(os.path.join(bc, 'edges.csv'))
    groups = pd.read_csv(os.path.join(bc, 'group-edges.csv'))
    num_nodes = max(edges.values.max(), groups['node_id'].max()) + 1
    x = torch.eye(num_nodes, dtype=torch.float)
    edge_index = torch.tensor(edges.values.T, dtype=torch.long)
    y = torch.zeros((num_nodes, groups['group_id'].max() + 1), dtype=torch.float)
    for _, row in groups.iterrows():
        y[row['node_id'], row['group_id']] = 1.0
    return wrap_tg_dataset('blogcatalog', Data(x=x, edge_index=edge_index, y=y))

def load_lastfm_asia(data_dir):
    base = os.path.join(data_dir, 'lastfm_asia')
    edges = pd.read_csv(os.path.join(base, 'lastfm_asia_edges.csv'))
    edge_index = torch.tensor(edges.values.T, dtype=torch.long)

    with open(os.path.join(base, 'lastfm_asia_features.json'), 'r') as f:
        feat_dict = json.load(f)
    features = pd.DataFrame.from_dict(feat_dict, orient='index')
    x = torch.tensor(features.values, dtype=torch.float)

    labels = pd.read_csv(os.path.join(base, 'lastfm_asia_target.csv'))
    y_arr = labels.values
    if y_arr.ndim == 2 and y_arr.shape[1] > 1:
        y = torch.argmax(torch.tensor(y_arr, dtype=torch.float), dim=1).long()
    else:
        y = torch.tensor(y_arr.squeeze(), dtype=torch.long)

    return wrap_tg_dataset('lastfm_asia', Data(x=x, edge_index=edge_index, y=y))

def load_dblp(data_dir):
    import numpy as np
    from scipy import sparse
    path = os.path.join(data_dir, 'dblp',  'dblp.npz')
    # path = os.path.join(data_dir, 'dblp', 'dblp', 'dblp.npz')

    with np.load(path, allow_pickle=True) as f:
        # adjacency
        adj = sparse.csr_matrix(
            (f['adj_data'], f['adj_indices'], f['adj_indptr']),
            shape=f['adj_shape']
        )
        # features
        feat = sparse.csr_matrix(
            (f['attr_data'], f['attr_indices'], f['attr_indptr']),
            shape=f['attr_shape']
        ).toarray()
        # labels
        labels = f['labels']

    edge_index = torch.tensor(np.vstack(adj.nonzero()), dtype=torch.long)
    x = torch.tensor(feat, dtype=torch.float)
    y = torch.tensor(labels.squeeze(), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    return wrap_tg_dataset('dblp', data)


def load_deezer_dataset(data_dir):
    import json, pandas as pd, torch, os
    base = os.path.join(data_dir, 'deezer')

    edges = pd.read_csv(os.path.join(base, 'DE_edges.csv'))
    edge_index = torch.tensor(edges.values.T, dtype=torch.long)

    with open(os.path.join(base, 'DE.json'), 'r') as f:
        feat_dict = json.load(f)
    features = pd.DataFrame.from_dict(feat_dict, orient='index')
    features = features.fillna(0)  
    x = torch.tensor(features.values, dtype=torch.float)
    labels = pd.read_csv(os.path.join(base, 'DE_target.csv')).astype(int)
    y = torch.tensor(labels.target.values.squeeze(), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    from torch_geometric.transforms import NormalizeFeatures
    data = NormalizeFeatures()(data)  

    return wrap_tg_dataset('deezer', data)

def load_pokec_mat(data_dir):
    """ requires pokec.mat """
    if not os.path.exists(f'{data_dir}/pokec/pokec.mat'):
        drive_id = '1575QYJwJlj7AWuOKMlwVmMz8FcslUncu'
        gdown.download(id=drive_id, output="data/pokec/")
        #import sys; sys.exit()
        #gdd.download_file_from_google_drive(
        #    file_id= drive_id, \
        #    dest_path=f'{data_dir}/pokec/pokec.mat', showsize=True)

    fulldata = scipy.io.loadmat(f'{data_dir}/pokec/pokec.mat')

    dataset = NCDataset('pokec')
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat']).float()
    num_nodes = int(fulldata['num_nodes'])
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}

    label = fulldata['label'].flatten()
    dataset.label = torch.tensor(label, dtype=torch.long)
    return dataset

# -------------------- dispatcher --------------------
# -------------------- GraphLand --------------------
# GraphLand lives at this repo path and is NOT a package; its module is also
# named dataset.py, so we load it by file under a distinct name to avoid
# colliding with this module. It resolves its own data dir via __file__.
_GRAPHLAND_ROOT = '/gfs/shared/public/datasets/graphland'
_GRAPHLAND_CANON = [
    'hm-categories', 'hm-prices', 'avazu-ctr', 'tolokers-2', 'artnet-views', 'artnet-exp',
    'twitch-views', 'city-roads-M', 'city-roads-L', 'city-reviews', 'pokec-regions',
    'web-fraud', 'web-traffic', 'web-topics',
]
_GRAPHLAND_MAP = {n.lower(): n for n in _GRAPHLAND_CANON}
_graphland_pygdataset = None


def _get_graphland_pygdataset():
    global _graphland_pygdataset
    if _graphland_pygdataset is None:
        path = os.path.join(_GRAPHLAND_ROOT, 'dataset.py')
        spec = importlib.util.spec_from_file_location('graphland_dataset', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _graphland_pygdataset = mod.PyGDataset
    return _graphland_pygdataset


def load_graphland(data_dir, name, split='RL'):
    if split.upper() == 'THI':
        raise NotImplementedError("GraphLand inductive split 'THI' returns 3 graphs; "
                                  "only transductive splits (RL/RH/TH) are supported here.")
    PyGDataset = _get_graphland_pygdataset()
    pygds = PyGDataset(name=name, split=split.upper())
    nan_label_nodes = torch.isnan(pygds[0].y)
    nan_train_mask = torch.isnan(pygds[0].train_mask)
    nan_val_mask = torch.isnan(pygds[0].val_mask)
    nan_test_mask = torch.isnan(pygds[0].test_mask)
    nan_x_mask = torch.isnan(pygds[0].x).any(dim=1)
    nan_mask = nan_label_nodes | nan_train_mask | nan_val_mask | nan_test_mask | nan_x_mask
    n_nodes = pygds[0].num_nodes
    if nan_mask.any():
        print(f"Warning: GraphLand dataset {name} has {nan_mask.sum().item()} nodes with NaN values "
              f"in labels, features, or masks; these nodes will be removed.")
        pygds[0].x = pygds[0].x[~nan_mask]
        pygds[0].y = pygds[0].y[~nan_mask]
        pygds[0].train_mask = pygds[0].train_mask[~nan_mask]
        pygds[0].val_mask = pygds[0].val_mask[~nan_mask]
        pygds[0].test_mask = pygds[0].test_mask[~nan_mask]
        # Reindex edge_index to remove edges connected to NaN nodes
        edge_index = pygds[0].edge_index
        mask_indices = torch.arange(n_nodes)[~nan_mask]
        index_map = torch.full((n_nodes,), -1, dtype=torch.long)
        index_map[mask_indices] = torch.arange(mask_indices.size(0))
        new_edge_index = index_map[edge_index]
        valid_edges = (new_edge_index >= 0).all(dim=0)
        pygds[0].edge_index = new_edge_index[:, valid_edges]
        
    ds = wrap_tg_dataset(name, pygds[0])  # transductive -> single Data
    return ds


def load_dataset(data_dir, name):
    # GraphLand datasets, optionally with a split suffix "name:SPLIT" (default RL).
    gl_name, _, gl_split = name.partition(':')
    if gl_name.lower() in _GRAPHLAND_MAP:
        return load_graphland(data_dir, _GRAPHLAND_MAP[gl_name.lower()], gl_split or 'RL')
    name = name.lower()
    if name in ('cora', 'citeseer', 'pubmed'):
        return load_planetoid(data_dir, name)
    if name in ('amazon-photo', 'amazon-computer'):
        return load_amazon(data_dir, name)
    if name in ('coauthor-cs', 'coauthor-physics'):
        return load_coauthor(data_dir, name)
    if name == 'wikics':
        return load_wikics(data_dir)
    if name == 'reddit':
        return load_reddit(data_dir)
    if name == 'cora-full':
        return load_cora_full(data_dir)
    if name in ('cornell', 'texas', 'wisconsin'):
        return load_webkb(data_dir, name)
    if name in ('chameleon'):
        return load_wikipedia_network(data_dir, name)
    if name in ('squirrel'):
        return load_wikipedia_network_squirrel(data_dir, name)
    if name in ('roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions'):
        return load_hetero(data_dir, name)
    if name in ('ogbn-arxiv', 'ogbn-products'):
        return load_ogb(data_dir, name)
    if name in ('ogbn-proteins'):
        return load_ogb_proteins(data_dir, name)
    if name == 'actor':
        return load_actor(data_dir)
    if name in ('air-brazil', 'air-eu', 'air-us', 'air-usa', 'air-europe'):
        return load_air(data_dir, name.split('-')[1])
    if name == 'blogcatalog':
        return load_blogcatalog(data_dir)
    if name == 'lastfm_asia':
        return load_lastfm_asia(data_dir)
    if name == 'dblp':
        return load_dblp(data_dir)
    if name == 'deezer':
        return load_deezer_europe(data_dir)
    if name == 'pokec':
        return load_pokec_mat(data_dir)
    raise ValueError(f"Unknown or unsupported dataset: {name}")
