import torch
import torch.nn as tnn
import torch_geometric as pyg
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch

import torch.nn.functional as F
from typing import Callable, Optional, Literal, Tuple

class MessageCAGCNLayer(pyg_nn.MessagePassing):
    """A message passing layer for Cross-Attention Graph Convolutional Networks (CAGCN).This version attends the messages before aggregation."""

    def __init__(
        self,
        in_channels: int,
        extra_features_channels: int,
        out_channels: int,
        num_heads: int,
        gcn_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        aggr: str = "add",
    ):
        super().__init__(aggr=aggr)

        # following the original formulation for GCN
        self.prj_messages = pyg_nn.Linear(in_channels, out_channels, bias=False)
        self.bias = tnn.Parameter(torch.zeros(out_channels))

        self.prj_extra_features = pyg_nn.Linear(
            extra_features_channels, out_channels
        )

        if out_channels % num_heads != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by num_heads ({num_heads})."
            )

        self.mha = tnn.MultiheadAttention(
            embed_dim=out_channels,
            num_heads=num_heads,
            batch_first=True,
            dropout=attn_dropout,
        )
        self.attn_weight = tnn.Parameter(torch.tensor(0.1))

        self.gcn_dropout = tnn.Dropout(gcn_dropout)

        self.reset_parameters()

    def reset_parameters(self):
        self.prj_messages.reset_parameters()
        self.bias.data.fill_(0)
        self.prj_extra_features.reset_parameters()
        self.attn_weight.data.fill_(0.1)

        # Reset MultiheadAttention parameters
        for module in self.mha.modules():
            if isinstance(module, tnn.Linear):
                tnn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    tnn.init.zeros_(module.bias)

    def forward(self, graph: Data, extra_features: torch.Tensor, extra_mask: torch.Tensor):
        """
        Forward pass of the MessageCAGCNLayer.

        Args:
            x (torch.Tensor): Node features of shape [num_nodes, in_channels].
            edge_index (torch.Tensor): Edge indices of shape [2, num_edges].
            extra_features (torch.Tensor): Extra features of shape [L, extra_features_channels].
        """

        x = graph.x
        edge_index = graph.edge_index
        batch = graph.batch if hasattr(graph, 'batch') else None

        messages = self.gcn_dropout(self.prj_messages(x))

        # Compute normalization
        row, col = edge_index
        deg = pyg.utils.degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Start propagating messages
        out = self.propagate(
            edge_index, x=messages, norm=norm, batch=batch, source_nodes=edge_index[0], extra_features=extra_features, extra_mask=extra_mask
        )

        # Apply a final bias vector
        out = out + self.bias

        return out

    def message(self, x_j, norm, batch, source_nodes, extra_features, extra_mask):
        """
        Message function for the CAGCN layer.

        Args:
            x_j (torch.Tensor): Messages from neighboring nodes, shape [num_edges, out_channels].
            norm (torch.Tensor): Normalization factors, shape [num_edges].
            extra_features (torch.Tensor): Extra features for attention, shape [B, L, extra_features_channels].

        Returns:
            torch.Tensor: Processed messages after attention.
        """

        x_j *= norm.view(-1, 1)  # GCN normalization

        _extra_features = self.gcn_dropout(
            self.prj_extra_features(extra_features)
        )

        messages_batch, messages_mask = to_dense_batch(x_j, batch[source_nodes])  # [num_graphs, max(num_edges), features], [num_graphs, max(num_edges)]

        # Use key_padding_mask to handle padding in extra_features (keys/values)
        # key_padding_mask should be True for positions that should be ignored
        key_padding_mask = ~extra_mask  # [B, S] - True for padded positions in extra_features

        attn_scores, _ = self.mha(
            query=messages_batch,
            key=_extra_features,
            value=_extra_features,
            key_padding_mask=key_padding_mask,  # Mask padded positions in keys/values
        )  # [B, max(num_edges), features]

        if attn_scores.isnan().any():
            raise ValueError("NaN detected in attention scores.")
        
        # Mask attention scores for padded message positions
        # This ensures that padded queries don't contribute any attention information
        attn_scores = attn_scores.masked_fill(~messages_mask.unsqueeze(-1), 0.0)
        
        # Undo the batching
        attn_scores = attn_scores[messages_mask]
        
        # Apply attention weight and add to original messages
        out = x_j + self.attn_weight * attn_scores
        
        return out


class MessageCAGCN(tnn.Module):

    def __init__(
        self,
        node_features_channels: int,
        extra_features_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_layers: int,
        attn_heads: int,
        activation: Callable = F.relu,
        gcn_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        embed_node_features: Literal["molecular", "linear", "none"] = "molecular",
        normalization: Literal["batch", "layer", "instance", "none"] = "none",
        input_node_features_channels: Optional[int] = None,
        *args,
        **kwargs
    ):

        super().__init__()

        self.layers = tnn.ModuleList()

        # First layer
        self.layers.append(
            MessageCAGCNLayer(
                in_channels=node_features_channels,
                extra_features_channels=extra_features_channels,
                out_channels=hidden_channels,
                num_heads=attn_heads,
                gcn_dropout=gcn_dropout,
                attn_dropout=attn_dropout,
            )
        )

        # Intermediate layers
        for _ in range(num_layers - 1):
            self.layers.append(
                MessageCAGCNLayer(
                    in_channels=hidden_channels,
                    extra_features_channels=extra_features_channels,
                    out_channels=hidden_channels,
                    num_heads=attn_heads,
                    gcn_dropout=gcn_dropout,
                    attn_dropout=attn_dropout,
                )
            )

        # project to output channels
        self.final_prj = pyg_nn.Linear(
            in_channels=hidden_channels,
            out_channels=out_channels,
            bias=True,
        )
        self.activation = activation

    def _resolve_normalization(
        self, normalization: str, num_features: int
    ) -> tnn.Module:
        """
        Resolve the normalization method based on the provided string.
        
        Args:
            normalization (str): Type of normalization ('batch', 'layer', 'instance', 'none').
            num_features (int): Number of features for the normalization layer.
            
        Returns:
            tnn.Module: The normalization layer.
        """
        if normalization == "batch":
            return tnn.BatchNorm1d(num_features)
        elif normalization == "layer":
            return tnn.LayerNorm(num_features)
        elif normalization == "instance":
            return tnn.InstanceNorm1d(num_features)
        elif normalization == "none":
            return tnn.Identity()
        else:
            raise ValueError(
                f"Invalid normalization: {normalization}. Choose from ['batch', 'layer', 'instance', 'none']."
            )

    def forward(self, graph: Data, extra_features: Tuple[torch.Tensor, torch.Tensor]):
        """
        Forward pass of the MessageCAGCN model.

        Args:
            graph (torch_geometric.data.Data): Graph data containing node features and edge index.
            extra_features (torch.Tensor): Extra features of shape [L, extra_features_channels].

        Returns:
            torch.Tensor: Output node representations after all layers.
        """
        # Unpack extra features
        extra_features, extra_mask = extra_features

        # Normalize extra features
        extra_features = F.layer_norm(extra_features, extra_features.shape[-1:])

        for i, layer in enumerate(self.layers):
            x = layer(graph, extra_features, extra_mask)
            x = F.layer_norm(x, x.shape[-1:])
            x = self.activation(x)
            if x.isnan().any():
                raise ValueError(f"NaN detected in node features after layer {i}.")
            graph.x = x

        # Last layer without activation
        x = self.final_prj(x)

        return x
    
class MCAMPNN(tnn.Module):

    def __init__(
        self,
        conv_cls: type,
        node_features_channels: int,
        extra_features_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_layers: int,
        attn_heads: int,
        edge_features_channels: int = 0,
        activation: Callable = F.relu,
        gcn_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        learned_scaler: bool = True,
        scaler_init: float = 0.1,
        embed_node_features: Literal["molecular", "linear", "embedding", "none"] = "molecular",
        embed_edge_features: Literal["molecular", "linear", "embedding", "none"] = "molecular",
        normalization: Literal["batch", "layer", "instance", "none"] = "none",
        input_node_features_channels: Optional[int] = None,
        input_edge_features_channels: Optional[int] = None,
        *args,
        **kwargs,
    ):

        super().__init__()

        self.embed_edges = lambda **kwargs: None  # No edge features, do nothing
        edge_dim = None

        self.layers = tnn.ModuleList()

        # First layer
        self.layers.append(
            conv_cls(
                in_channels=node_features_channels,
                out_channels=hidden_channels,
                mca_kv_dim=extra_features_channels,
                mca_num_heads=attn_heads,
                mca_dropout=attn_dropout,
                scaler_learned=learned_scaler,
                scaler_init=scaler_init,
                edge_dim=edge_dim,
            )
        )

        # Intermediate layers
        for _ in range(num_layers - 1):
            self.layers.append(
                conv_cls(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    mca_kv_dim=extra_features_channels,
                    mca_num_heads=attn_heads,
                    mca_dropout=attn_dropout,
                    scaler_learned=learned_scaler,
                    scaler_init=scaler_init,
                    edge_dim=edge_dim,
                )
            )

        # project to output channels
        self.final_prj = pyg_nn.Linear(
            in_channels=hidden_channels,
            out_channels=out_channels,
            bias=True,
        )
        self.activation = activation

    def _resolve_normalization(
        self, normalization: str, num_features: int
    ) -> tnn.Module:
        """
        Resolve the normalization method based on the provided string.

        Args:
            normalization (str): Type of normalization ('batch', 'layer', 'instance', 'none').
            num_features (int): Number of features for the normalization layer.

        Returns:
            tnn.Module: The normalization layer.
        """
        if normalization == "batch":
            return tnn.BatchNorm1d(num_features)
        elif normalization == "layer":
            return tnn.LayerNorm(num_features)
        elif normalization == "instance":
            return tnn.InstanceNorm1d(num_features)
        elif normalization == "none":
            return tnn.Identity()
        else:
            raise ValueError(
                f"Invalid normalization: {normalization}. Choose from ['batch', 'layer', 'instance', 'none']."
            )

    def forward(
        self,
        graph: Data,
        extra_features: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the MCAMPNN model. This implementation disregards edge features.

        Args:
            graph (Data): Graph data containing node features and edge index.
            extra_features (Tuple[torch.Tensor, torch.Tensor]): Extra features and mask.

        Returns:
            Tensor: Output node representations.
        """
        # Unpack extra features
        extra_features, extra_mask = extra_features
        # Extract node features, edge indices, and batch information
        x = graph.x
        edge_index = graph.edge_index
            
        batch = getattr(graph, 'batch', None)
        
        # For graph level tasks, different graphs are batched together and 
        # identified by 'batch' attribute. If not present, assume all nodes 
        # belong to a single graph.
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # remove unneeded dimension added by nn.Embedding
        if x.dim() == 3:
            x = x.squeeze(1)  

        # Handle edge features if present
        edge_attr = None

        # Normalize extra features
        extra_features = F.layer_norm(extra_features, extra_features.shape[-1:])

        for layer in self.layers:
            x = layer(x, 
                      edge_index, 
                      graphs_batch=batch, 
                      edge_attr=edge_attr,
                      extra_kv=extra_features, 
                      extra_kv_mask=extra_mask)
            x = F.layer_norm(x, x.shape[-1:])
            x = self.activation(x)

        # Last layer without activation, update graph node representations
        x = self.final_prj(x)

        return x
