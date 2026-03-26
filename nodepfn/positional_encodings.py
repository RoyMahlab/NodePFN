import math

import torch
from torch import nn

from torch_geometric.utils import get_laplacian, to_dense_adj

# Protocol for positonal encodings.
# __init__(d_model, max_len=..[, more optionals])
# forward(x: (seq_len, bs, d_model)) -> Tensor of shape (*x.shape[:2],d_model) containing pos. embeddings


class NoPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=None):
        super(NoPositionalEncoding, self).__init__()
        pass

    def forward(self, x):
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = self.pe[:x.size(0), :] + x # * math.sqrt(x.shape[-1])
        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(LearnedPositionalEncoding, self).__init__()
        self.max_seq_len = max_len
        #self.positional_embeddings = nn.Embedding(max_len, d_model)
        self.positional_embeddings = nn.Parameter(torch.empty(max_len, d_model))
        nn.init.normal_(self.positional_embeddings, mean=0, std=d_model ** -0.5)

    def forward(self, x):
        seq_len, bs, d_model = x.shape
        assert seq_len <= len(self.positional_embeddings), 'seq_len can be at most max_len.'
        pos_emb = self.positional_embeddings[:seq_len]
        return pos_emb.unsqueeze(1).expand(seq_len, bs, d_model) + x #* math.sqrt(x.shape[-1])


class PairedScrambledPositionalEncodings(LearnedPositionalEncoding):
    # TODO check whether it is a problem to use the same perm. for full batch
    def forward(self, x):
        seq_len, bs, d_model = x.shape
        assert seq_len <= len(self.positional_embeddings), 'seq_len can be at most max_len.'
        assert len(self.positional_embeddings) % 2 == 0, 'Please specify an even max_len.'

        paired_embs = self.positional_embeddings.view(len(self.positional_embeddings), -1, 2)
        pos_emb = paired_embs[torch.randperm(len(paired_embs))].view(*self.positional_embeddings.shape)[:seq_len]

        return pos_emb.unsqueeze(1).expand(seq_len, bs, d_model) + x #* math.sqrt(x.shape[-1])


class LapPePositionalEncodings(nn.Module):
    """
    Laplacian Positional Encoding for graphs using eigenvectors of the Laplacian matrix.
    Based on "A Generalization of Transformer to Graphs" (Dwivedi & Bresson, 2020).
    """
    def __init__(self, d_model, max_len=5000, k_eigenvectors=8, normalization='sym'):
        super(LapPePositionalEncodings, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.k_eigenvectors = k_eigenvectors
        self.normalization = normalization
        
        # Linear layer to project eigenvectors to d_model dimension
        self.linear = nn.Linear(k_eigenvectors, d_model)
        
    def compute_laplacian_pe(self, edge_index, num_nodes, batch_size=None):
        """
        Compute Laplacian positional encoding for a graph.
        
        Args:
            edge_index: Edge indices in COO format [2, num_edges]
            num_nodes: Number of nodes in the graph
            batch_size: Batch size for expanding encodings
        
        Returns:
            Laplacian positional encodings [num_nodes, d_model]
        """
        device = edge_index.device
        
        # Compute Laplacian matrix
        edge_index_laplacian, edge_weight = get_laplacian(
            edge_index, num_nodes=num_nodes, normalization=self.normalization
        )
    
        # Convert to dense adjacency matrix
        L = to_dense_adj(
            edge_index_laplacian, edge_attr=edge_weight, max_num_nodes=num_nodes
        ).squeeze(0)  # [num_nodes, num_nodes]
        
        # Compute eigenvalues and eigenvectors
        try:
            eigenvals, eigenvecs = torch.linalg.eigh(L)
            
            # Sort by eigenvalues (ascending)
            idx = eigenvals.argsort()
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            # Take the smallest k eigenvectors (excluding the first constant eigenvector)
            # Start from index 1 to skip the constant eigenvector
            start_idx = 1 if eigenvals[0].abs() < 1e-6 else 0
            end_idx = min(start_idx + self.k_eigenvectors, eigenvecs.size(1))
            eigenvecs = eigenvecs[:, start_idx:end_idx]
            
            # Pad with zeros if we don't have enough eigenvectors
            if eigenvecs.size(1) < self.k_eigenvectors:
                padding = torch.zeros(num_nodes, self.k_eigenvectors - eigenvecs.size(1), 
                                    device=device, dtype=eigenvecs.dtype)
                eigenvecs = torch.cat([eigenvecs, padding], dim=1)
                
        except Exception as e:
            print(f"Warning: Failed to compute eigendecomposition, using random encoding: {e}")
            eigenvecs = torch.randn(num_nodes, self.k_eigenvectors, device=device)
        
        # Project to d_model dimension
        pe = self.linear(eigenvecs)  # [num_nodes, d_model]
        
        return pe
        
    def forward(self, x, edge_index=None, batch=None):
        """
        Forward pass for Laplacian positional encoding.
        
        Args:
            x: Node features [seq_len, bs, d_model] or [num_nodes, d_model]
            edge_index: Edge indices [2, num_edges] (required for computing Laplacian PE)
            batch: Batch vector for batched graphs (optional)
        
        Returns:
            x + Laplacian PE
        """
        if edge_index is None:
            raise ValueError("edge_index is required for LaplacianPositionalEncoding")
            
        # Handle different input shapes
        if x.dim() == 3:
            seq_len, bs, d_model = x.shape
            num_nodes = seq_len
            # Compute Laplacian PE
            if len(edge_index) == 0:
                return x
            else:
                pe = self.compute_laplacian_pe(edge_index, num_nodes, bs)
            
                # Expand for batch dimension
                pe = pe.unsqueeze(1).expand(seq_len, bs, d_model)
                return x + pe
                
        elif x.dim() == 2:
            num_nodes, d_model = x.shape
            if len(edge_index) == 0:
                return x
            else:
                pe = self.compute_laplacian_pe(edge_index, num_nodes)
                return x + pe


class RandomWalkStructuralEncoding(nn.Module):
    """
    Random Walk Structural Encoding for graphs.
    Encodes structural information using random walk statistics.
    """
    def __init__(self, d_model, max_len=5000, walk_length=16, num_walks=1):
        super(RandomWalkStructuralEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.walk_length = walk_length
        self.num_walks = num_walks
        
        # Linear layers to process random walk features
        self.rw_linear = nn.Linear(walk_length, d_model // 2)
        self.landing_prob_linear = nn.Linear(walk_length, d_model // 2)
        
        # For caching computed random walk features
        self._cached_rw_features = {}
        
    def compute_random_walk_features(self, edge_index, num_nodes):
        """
        Compute random walk structural features.
        
        Args:
            edge_index: Edge indices [2, num_edges]
            num_nodes: Number of nodes
            
        Returns:
            Random walk features [num_nodes, walk_length * 2]
        """
        device = edge_index.device
        
        # Create cache key
        cache_key = tuple(edge_index.cpu().numpy().flatten()) + (num_nodes,)
        
        if cache_key in self._cached_rw_features:
            return self._cached_rw_features[cache_key].to(device)
            
        # Convert to adjacency matrix
        if HAS_TORCH_GEOMETRIC:
            adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
        else:
            # Fallback implementation
            adj = torch.zeros(num_nodes, num_nodes, device=device)
            adj[edge_index[0], edge_index[1]] = 1.0
            
        # Add self-loops for stability
        adj = adj + torch.eye(num_nodes, device=device)
        
        # Normalize adjacency matrix (row-wise)
        row_sum = adj.sum(dim=1, keepdim=True)
        row_sum[row_sum == 0] = 1  # Avoid division by zero
        transition_matrix = adj / row_sum
        
        # Compute random walk features
        rw_features = []
        landing_probs = []
        
        current_prob = torch.eye(num_nodes, device=device)  # Start from each node
        
        for step in range(self.walk_length):
            # Random walk return probability (diagonal elements)
            rw_return_prob = torch.diag(current_prob)
            rw_features.append(rw_return_prob.unsqueeze(1))
            
            # Landing probability after k steps (sum of probabilities)
            landing_prob = current_prob.sum(dim=1)
            landing_probs.append(landing_prob.unsqueeze(1))
            
            # Update probabilities for next step
            current_prob = torch.mm(current_prob, transition_matrix)
            
        # Concatenate features across steps
        rw_features = torch.cat(rw_features, dim=1)  # [num_nodes, walk_length]
        landing_probs = torch.cat(landing_probs, dim=1)  # [num_nodes, walk_length]
        
        # Cache the results
        features = torch.cat([rw_features, landing_probs], dim=1)
        self._cached_rw_features[cache_key] = features.cpu()
        
        return features
        
    def forward(self, x, edge_index=None, batch=None):
        """
        Forward pass for Random Walk Structural Encoding.
        
        Args:
            x: Node features [seq_len, bs, d_model] or [num_nodes, d_model]
            edge_index: Edge indices [2, num_edges]
            batch: Batch vector for batched graphs (optional)
            
        Returns:
            x + Random Walk SE
        """
        if edge_index is None:
            raise ValueError("edge_index is required for RandomWalkStructuralEncoding")
            
        # Handle different input shapes
        if x.dim() == 3:
            seq_len, bs, d_model = x.shape
            num_nodes = seq_len
            
            # Compute random walk features
            rw_features = self.compute_random_walk_features(edge_index, num_nodes)
            rw_part = rw_features[:, :self.walk_length]  # [num_nodes, walk_length]
            landing_part = rw_features[:, self.walk_length:]  # [num_nodes, walk_length]
            
            # Project to embedding space
            rw_emb = self.rw_linear(rw_part)  # [num_nodes, d_model//2]
            landing_emb = self.landing_prob_linear(landing_part)  # [num_nodes, d_model//2]
            
            # Concatenate embeddings
            pe = torch.cat([rw_emb, landing_emb], dim=1)  # [num_nodes, d_model]
            
            # Expand for batch dimension
            pe = pe.unsqueeze(1).expand(seq_len, bs, d_model)
            
        elif x.dim() == 2:
            num_nodes, d_model = x.shape
            
            # Compute random walk features
            rw_features = self.compute_random_walk_features(edge_index, num_nodes)
            rw_part = rw_features[:, :self.walk_length]
            landing_part = rw_features[:, self.walk_length:]
            
            # Project to embedding space
            rw_emb = self.rw_linear(rw_part)
            landing_emb = self.landing_prob_linear(landing_part)
            
            # Concatenate embeddings
            pe = torch.cat([rw_emb, landing_emb], dim=1)
            
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
            
        return x + pe