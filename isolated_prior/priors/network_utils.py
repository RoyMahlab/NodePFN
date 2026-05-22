import torch
from torch import nn

import networkx as nx
import numpy as np

def generate_edge_index(x, y, h, device):
    """
    x shape: (T, B, H) - T: seq_len, B: batch_size, H: num_features
    y shape: (T, B)
    """
    
    for b in range(x.shape[1]):  
        num_nodes = x.shape[0] 
        
        if h['graph_type'] == 'random':
            # Erdős-Rényi random graph
            edge_prob = h['edge_prob']
            G = nx.erdos_renyi_graph(num_nodes, edge_prob)
            
        elif h['graph_type'] == 'sbm':
            # Stochastic Block Model
            node_labels = y[:, b].long().cpu().numpy()
            num_communities = len(torch.unique(y[:, b]))
            
            sizes = [int((node_labels == i).sum()) for i in range(num_communities)]
            
            homophily_rate = h['homophily_rate']
            p_in = h['p_in']
            p_out = p_in * (1 - homophily_rate)

            probs = np.random.power(5, (num_communities, num_communities)) * p_out

            diagonal_values = p_out + np.random.power(2, num_communities) * (p_in - p_out)
            np.fill_diagonal(probs, diagonal_values)

            probs = (probs + probs.T) / 2

            node_mapping = []
            for comm in range(num_communities):
                node_mapping.extend(np.where(node_labels == comm)[0])
            
            G_temp = nx.stochastic_block_model(sizes, probs)
            
            G = nx.Graph()
            G.add_nodes_from(range(num_nodes))
            
            for i, j in G_temp.edges():
                G.add_edge(node_mapping[i], node_mapping[j])
        
        edges = list(G.edges())
        if len(edges) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        else:
            edge_index = torch.tensor(edges, device=device, dtype=torch.long).t()
    return edge_index