from torch import nn as tnn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric import nn as pygnn
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Data
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.typing import Size
from torch_geometric.nn.models import MLP
from torch_geometric.nn.conv import GCNConv, SAGEConv, GATConv, GATv2Conv, GINConv, GraphConv, GPSConv, GINEConv
from typing import Optional, Tuple, Union, List
import torch


def resolve_target_backbone(backbone_name: str, is_vanilla: bool):
    """
    Resolve and return the appropriate graph neural network layer class based on backbone name and variant.
    Args:
        backbone_name (str): The name of the backbone architecture. Supported values are:
            "gcn", "graphsage", "gat", "gatv2", "gin", "graphconv", "gps"
        is_vanilla (bool): If True, returns the vanilla implementation of the layer.
                          If False, returns the MCA (Multi-Channel Attention) variant.
    Returns:
        class: The corresponding layer class (either vanilla or MCA variant)
    Raises:
        AssertionError: If the provided backbone_name is not supported
    Examples:
        >>> resolve_target_backbone("gcn", True)
        <class 'GCNConv'>
        >>> resolve_target_backbone("gcn", False)
        <class 'MCAGCNConv'>
    """

    name2layer_ours = {
        "gcn": MCAGCNConv,
        "graphsage": MCASageConv,
        "gat": MCAGATConv,
        "gatv2": MCAGATv2Conv,
        "gin": MCAGINConv,
        "gine": MCAGINEConv,
        "graphconv": MCAGraphConv,
        "gps": MCAGPSConv,
    }
    name2layer_vanilla = {
        "gcn": GCNConv,
        "graphsage": SAGEConv,
        "gat": lambda in_channels, out_channels, edge_dim=None, **kwargs: GATConv(
            in_channels, out_channels, edge_dim=edge_dim, add_self_loops=False, **kwargs
        ),
        "gatv2": lambda in_channels, out_channels, edge_dim=None, **kwargs: GATv2Conv(
            in_channels, out_channels, edge_dim=edge_dim, add_self_loops=False, **kwargs
        ),
        "gin": lambda in_channels, out_channels, **kwargs: GINConv(
            MLP([in_channels, out_channels, out_channels]), **kwargs
        ),
        "gine": lambda in_channels, out_channels, edge_dim=None, **kwargs: GINEConv(
            MLP([in_channels, out_channels, out_channels]),
            edge_dim=edge_dim,
            **kwargs,
        ),
        "graphconv": GraphConv,
        "gps": lambda in_channels, out_channels, **kwargs: GPSConv(
            channels=out_channels,
            conv=GINConv(MLP([in_channels, out_channels, out_channels])),
            **kwargs,
        ),
    }

    assert (
        backbone_name in name2layer_ours
    ), f"Backbone {backbone_name} not supported."

    return (
        name2layer_vanilla[backbone_name]
        if is_vanilla
        else name2layer_ours[backbone_name]
    )

class CrossAttentionLayer(tnn.Module):

    def __init__(self, embed_dim: int, kv_dim: int, num_heads: int, dropout: float = 0.0, scaler_learned: bool = True, scaler_init: float = 0.1):
        """Initialize CrossAttentionLayer.

        Args:
            embed_dim (int): The embedding dimension for queries and output projections.
            kv_dim (int): The dimension of key and value inputs.
            num_heads (int): Number of parallel attention heads.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            scaler_learned (bool, optional): Whether the cross-attention scaling factor is learnable.
                If True, ca_scaler is a learnable parameter updated during training.
                If False, ca_scaler is a fixed buffer that does not receive gradients. Defaults to True.
            scaler_init (float, optional): Initial value for the cross-attention scaling factor. Defaults to 0.1.
        """
        super(CrossAttentionLayer, self).__init__()
        self.cross_attention = tnn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = tnn.Dropout(dropout)
        self.prj_kv = pygnn.Linear(kv_dim, embed_dim)
        
        if scaler_learned:
            self.ca_scaler = tnn.Parameter(torch.tensor(scaler_init))
        else:
            self.register_buffer('ca_scaler', torch.tensor(scaler_init))

    def reset_parameters(self):
        """Reinitialize learnable parameters."""

        for module in self.modules():
            if isinstance(module, tnn.Linear):
                tnn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    tnn.init.zeros_(module.bias)

    def forward(self, messages: torch.Tensor, messages_batch: torch.Tensor, kv: torch.Tensor, kv_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for cross-attention mechanism between messages and key-value pairs.
        Args:
            messages: Input message embeddings to be used as queries. Shape: [num_messages, embed_dim]
            messages_batch: Batch information for the messages, associating each message to its graph in the batch. Shape: [num_messages]
            kv: Key-value embeddings to attend to. Shape: [batch_size, length, kv_dim]
            kv_mask (optional): Mask for key-value pairs. True indicates valid positions,
                               False indicates padding positions that should be ignored. Shape: [batch_size, length]
        Returns:
            torch.Tensor: Output embeddings with residual connection and scaled cross-attention. Shape: [num_messages, embed_dim]
                         applied to the input messages
        Process:
        """

        kv = self.prj_kv(kv)  # Project kv to embed_dim

        if kv_mask is not None:
            kv_mask = ~kv_mask  # Invert mask: True for padding positions
        q_batch, q_mask = to_dense_batch(messages, messages_batch)

        attn_output, _ = self.cross_attention(
            query=q_batch,
            key=kv,
            value=kv,
            key_padding_mask=kv_mask)
        attn_output = self.dropout(attn_output)

        # Mask attention output for padded message positions
        # This ensures that padded queries don't contribute any attention information
        attn_output = attn_output.masked_fill(~q_mask.unsqueeze(-1), 0.0)

        # Undo the batching
        attn_output = attn_output[q_mask]

        # Residual connection with scaling
        return messages + self.ca_scaler * attn_output  


class MCAGATConv(pygnn.conv.GATConv):
    """
    Multi-Channel Attention Graph Attention Network Convolution Layer.
    This layer extends the standard GAT convolution by incorporating a cross-attention mechanism
    that allows passed messages to attend to additional key-value embeddings.
    """
    
    # Used to regulate reset_parameters behavior
    __init = False
    
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        mca_kv_dim: int,
        mca_num_heads: int,
        mca_dropout: float = 0.0,
        scaler_learned: bool = True,
        scaler_init: float = 0.1,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = False,  # We add self-loops (if needed) during data preprocessing
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        residual: bool = False,
        **kwargs,
    ):
        
        super(MCAGATConv, self).__init__(
            in_channels,
            out_channels,
            heads=heads,
            concat=concat,
            negative_slope=negative_slope,
            dropout=dropout,
            add_self_loops=add_self_loops,
            edge_dim=edge_dim,
            fill_value=fill_value,
            bias=bias,
            residual=residual,
            **kwargs,
        )
        
        # Cross-attention layer operates on messages with all heads (heads * out_channels)
        mca_embed_dim = heads * out_channels
        self.mca_layer = CrossAttentionLayer(
            embed_dim=mca_embed_dim,
            kv_dim=mca_kv_dim,
            num_heads=mca_num_heads,
            dropout=mca_dropout,
            scaler_learned=scaler_learned,
            scaler_init=scaler_init
        )
        
        self.__init = True
        
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        super(MCAGATConv, self).reset_parameters()
        
        # Only reset MCA parameters if this is not called from the parent class during __init__
        if self.__init:
            self.mca_layer.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        graphs_batch: Optional[Tensor] = None,
        extra_kv: Optional[Tensor] = None,
        extra_kv_mask: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
        return_attention_weights: bool = False,
        **kwargs,
    ) -> Tensor:
        """
        This is a modified version of the original Torch Geometric implementation that
        accepts PyG Data objects directly and supports additional key-value features
        that are passed to the message method.
        Args:
            x (Tensor): Node features.
            edge_index (Tensor): Graph connectivity in COO format.
            batch (Optional[Tensor], optional): Batch vector, where each element
                                                indicates the graph a node belongs to.
                                                Defaults to None.
            extra_kv (Optional[Tensor], optional): Additional key-value features to be used in
                                                   message passing.
            extra_kv_mask (Optional[Tensor], optional): Mask for the extra key-value features.
                                                         Defaults to None.
            edge_attr (Optional[Tensor], optional): Edge attribute tensor to be used by the
                                                     underlying GAT/GATv2 layer when
                                                     configured with `edge_dim`. Shape is
                                                     usually [num_edges, edge_dim]. Defaults to None.
            return_attention_weights (bool, optional): Whether to return attention weights.
                                                       Defaults to False.
            **kwargs: Additional arguments passed to the parent GATConv.
        Returns:
            Tensor: Updated node representations after message passing and cross-attention enhancement.
        """
        
        self._extra_kv = extra_kv
        self._extra_kv_mask = extra_kv_mask
        self._x_j_batch = graphs_batch[edge_index[1]]

        # Call parent GAT forward method (this will call our overridden message method)
        result = super(MCAGATConv, self).forward(
            x, edge_index, edge_attr=edge_attr, **kwargs
        )
        
        # If not returning attention weights, extract the tensor from the tuple
        if not return_attention_weights and isinstance(result, tuple):
            result = result[0]
        
        # Clean up stored variables
        self._extra_kv = None
        self._extra_kv_mask = None
        self._x_j_batch = None
        
        return result
    
    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        """
        Computes messages for graph convolution with multi-head cross-attention.
        This method extends the standard GAT message passing by incorporating
        cross-attention with extra key-value embeddings. The original message passing
        mechanism remains unchanged - this implementation simply adds cross-attention
        with extra features on top of the base functionality.
        Args:
            x_j (Tensor): Node features of neighboring nodes after GAT attention.
            alpha (Tensor): GAT attention coefficients.
        Returns:
            Tensor: Enhanced node messages after applying cross-attention with extra features.
        """
        
        # Standard GAT message passing
        x_j = super(MCAGATConv, self).message(x_j, alpha)

        # Cross-Attention with extra key-value embeddings
        if self._extra_kv is not None:
            x_j = self.mca_layer(x_j.squeeze(1), self._x_j_batch, self._extra_kv, self._extra_kv_mask).unsqueeze(1)

        return x_j


class MCAGATv2Conv(pygnn.conv.GATv2Conv):
    """
    Multi-Channel Attention GATv2 Convolution Layer.
    This layer extends the standard GATv2 convolution by incorporating a cross-attention mechanism
    that allows passed messages to attend to additional key-value embeddings.
    """
    
    # Used to regulate reset_parameters behavior
    __init = False
    
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        mca_kv_dim: int,
        mca_num_heads: int,
        mca_dropout: float = 0.0,
        scaler_learned: bool = True,
        scaler_init: float = 0.1,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = False,  # We add self-loops (if needed) during data preprocessing
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        share_weights: bool = False,
        residual: bool = False,
        **kwargs,
    ):
        
        super(MCAGATv2Conv, self).__init__(
            in_channels,
            out_channels,
            heads=heads,
            concat=concat,
            negative_slope=negative_slope,
            dropout=dropout,
            add_self_loops=add_self_loops,
            edge_dim=edge_dim,
            fill_value=fill_value,
            bias=bias,
            share_weights=share_weights,
            residual=residual,
            **kwargs,
        )
        
        # Cross-attention layer operates on messages with all heads (heads * out_channels)
        mca_embed_dim = heads * out_channels
        self.mca_layer = CrossAttentionLayer(
            embed_dim=mca_embed_dim,
            kv_dim=mca_kv_dim,
            num_heads=mca_num_heads,
            dropout=mca_dropout,
            scaler_learned=scaler_learned,
            scaler_init=scaler_init
        )
        
        self.__init = True
        
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        super(MCAGATv2Conv, self).reset_parameters()
        
        # Only reset MCA parameters if this is not called from the parent class during __init__
        if self.__init:
            self.mca_layer.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        graphs_batch: Optional[Tensor] = None,
        extra_kv: Optional[Tensor] = None,
        extra_kv_mask: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
        return_attention_weights: bool = False,
        **kwargs,
    ) -> Tensor:
        """
        This is a modified version of the original Torch Geometric implementation that
        accepts PyG Data objects directly and supports additional key-value features
        that are passed to the message method.
        """
        
        # Store extra key-value features and batch info for use in message function
        self._extra_kv = extra_kv
        self._extra_kv_mask = extra_kv_mask
        self._x_j_batch = graphs_batch[edge_index[1]]

        # Call parent GATv2Conv forward method (this will call our overridden message method)
        result = super(MCAGATv2Conv, self).forward(
            x, edge_index, edge_attr=edge_attr, **kwargs
        )

        # If not returning attention weights, extract the tensor from the tuple
        if not return_attention_weights and isinstance(result, tuple):
            result = result[0]

        # Clean up stored variables
        self._extra_kv = None
        self._extra_kv_mask = None
        self._x_j_batch = None
        
        return result
    
    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        """
        Computes messages for graph convolution with multi-head cross-attention.
        This method extends the standard GATv2 message passing by incorporating
        cross-attention with extra key-value embeddings.
        """
        
        # Standard GATv2 message passing
        x_j = super(MCAGATv2Conv, self).message(x_j, alpha)
        
        # Cross-Attention with extra key-value embeddings
        if self._extra_kv is not None:
            x_j = self.mca_layer(x_j.squeeze(1), 
                                 self._x_j_batch, 
                                 self._extra_kv, 
                                 self._extra_kv_mask).unsqueeze(1)


        return x_j


class MCAGraphConv(pygnn.conv.GraphConv):
    """
    Multi-Channel Attention Graph Convolution Layer.
    This layer extends the standard Graph convolution by incorporating a cross-attention mechanism
    that allows passed messages to attend to additional key-value embeddings.
    """
    
    # Used to regulate reset_parameters behavior
    __init = False
    
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        mca_kv_dim: int,
        mca_num_heads: int,
        mca_dropout: float = 0.0,
        scaler_learned: bool = True,
        scaler_init: float = 0.1,
        aggr: str = 'add',
        bias: bool = True,
        edge_dim: Optional[int] = None,
        **kwargs,
    ):
        
        super(MCAGraphConv, self).__init__(
            in_channels,
            out_channels,
            aggr=aggr,
            bias=bias,
            **kwargs,
        )
        
        self.mca_layer = CrossAttentionLayer(
            embed_dim=in_channels,  # in GraphConv, messages have in_channels dimension
            kv_dim=mca_kv_dim,
            num_heads=mca_num_heads if in_channels != 1 else 1,  # Ensure valid number of heads
            dropout=mca_dropout,
            scaler_learned=scaler_learned,
            scaler_init=scaler_init
        )
        
        self.__init = True
        
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        super(MCAGraphConv, self).reset_parameters()
        
        # Only reset MCA parameters if this is not called from the parent class during __init__
        if self.__init:
            self.mca_layer.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        graphs_batch: Optional[Tensor] = None,
        extra_kv: Optional[Tensor] = None,
        extra_kv_mask: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        This is a modified version of the original Torch Geometric implementation that
        accepts PyG Data objects directly and supports additional key-value features
        that are passed to the message method.
        """
        
        # Store extra key-value features and batch info for use in message function
        self._extra_kv = extra_kv
        self._extra_kv_mask = extra_kv_mask
        self._x_j_batch = graphs_batch[edge_index[1]]  # Batch info for target nodes of each edge

        # Call parent GraphConv forward method (this will call our overridden message method)
        # Pass edge_attr explicitly so message() receives it when the parent supports it.
        # Assumes edge_attr is actually a scalar edge weight tensor.
        result = super(MCAGraphConv, self).forward(x, edge_index, **kwargs)

        # Clean up stored variables
        self._extra_kv = None
        self._extra_kv_mask = None
        self._x_j_batch = None

        return result
    
    def message(self, x_j: Tensor, edge_weight: Optional[Tensor]) -> Tensor:
        """
        Computes messages for graph convolution with multi-head cross-attention.
        This method extends the standard Graph convolution message passing by incorporating
        cross-attention with extra key-value embeddings.
        """
        
        # Standard Graph message passing
        x_j = super(MCAGraphConv, self).message(x_j, edge_weight)

        # Cross-Attention with extra key-value embeddings
        if self._extra_kv is not None:
            x_j = self.mca_layer(x_j, self._x_j_batch, self._extra_kv, self._extra_kv_mask)

        return x_j


class MCASageConv(pygnn.conv.SAGEConv):
    """
    Multi-Channel Attention GraphSAGE Convolution Layer.
    This layer extends the standard GraphSAGE convolution by incorporating a cross-attention mechanism
    that allows passed messages to attend to additional key-value embeddings.
    """
    
    # Used to regulate reset_parameters behavior
    __init = False
    
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        mca_kv_dim: int,
        mca_num_heads: int,
        mca_dropout: float = 0.0,
        scaler_learned: bool = True,
        scaler_init: float = 0.1,
        aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
        edge_dim: Optional[int] = None,
        **kwargs,
    ):
        
        super(MCASageConv, self).__init__(
            in_channels,
            out_channels,
            aggr=aggr,
            normalize=normalize,
            root_weight=root_weight,
            project=project,
            bias=bias,
            **kwargs,
        )
        self.mca_layer = CrossAttentionLayer(
            embed_dim=in_channels,  # in GraphSAGE, messages have in_channels dimension
            kv_dim=mca_kv_dim,
            num_heads=mca_num_heads if in_channels != 1 else 1,  # Ensure valid number of heads,
            dropout=mca_dropout,
            scaler_learned=scaler_learned,
            scaler_init=scaler_init
        )
        
        self.__init = True
        
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        super(MCASageConv, self).reset_parameters()
        
        # Only reset MCA parameters if this is not called from the parent class during __init__
        if self.__init:
            self.mca_layer.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        graphs_batch: Optional[Tensor] = None,
        extra_kv: Optional[Tensor] = None,
        extra_kv_mask: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        This is a modified version of the original Torch Geometric implementation that
        accepts PyG Data objects directly and supports additional key-value features
        that are passed to the message method.
        Args:
            x (Tensor): Node features.
            edge_index (Tensor): Graph connectivity in COO format.
            batch (Optional[Tensor], optional): Batch vector, where each element
                                                indicates the graph a node belongs to.
                                                Defaults to None.
            extra_kv (Optional[Tensor], optional): Additional key-value features to be used in
                                                   message passing.
            extra_kv_mask (Optional[Tensor], optional): Mask for the extra key-value features.
                                                         Defaults to None.
            **kwargs: Additional arguments passed to the parent SAGEConv.
        Returns:
            Tensor: Updated node representations after message passing and linear
                    transformations.
        """
        
        # Store extra key-value features and batch info for use in message function
        self._extra_kv = extra_kv
        self._extra_kv_mask = extra_kv_mask
        self._x_j_batch = graphs_batch[edge_index[1]]

        # Call parent SAGEConv forward method (this will call our overridden message method)
        result = super(MCASageConv, self).forward(x, edge_index, **kwargs)
        
        # Clean up stored variables
        self._extra_kv = None
        self._extra_kv_mask = None
        self._x_j_batch = None
        
        return result
    
    def message(self, x_j):
        """
        Computes messages for graph convolution with multi-head cross-attention.
        This method extends the standard GraphSAGE message passing by incorporating
        cross-attention with extra key-value embeddings. The original message passing
        mechanism remains unchanged - this implementation simply adds cross-attention
        with extra features on top of the base functionality.
        Args:
            x_j (Tensor): Node features of neighboring nodes.
        Returns:
            Tensor: Enhanced node messages after applying cross-attention with extra features.
        """
        
        # Standard GraphSAGE message passing
        x_j = super(MCASageConv, self).message(x_j) 

        # Cross-Attention with extra key-value embeddings
        if self._extra_kv is not None:
            x_j = self.mca_layer(x_j, self._x_j_batch, self._extra_kv, self._extra_kv_mask)

        return x_j


class MCAGCNConv(pygnn.conv.GCNConv):
    """
    Multi-Channel Attention Graph Convolutional Network Convolution Layer.
    This layer extends the standard GCN convolution by incorporating a cross-attention mechanism
    that allows passed messages to attend to additional key-value embeddings.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mca_kv_dim: int,
        mca_num_heads: int,
        mca_dropout: float = 0.0,
        scaler_learned: bool = True,
        scaler_init: float = 0.1,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = False,  # We add self-loops (if needed) during data preprocessing
        normalize: bool = True,
        bias: bool = True,
        edge_dim: Optional[int] = None,
        **kwargs,
    ):
        
        super(MCAGCNConv, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            improved=improved,
            cached=cached,
            add_self_loops=add_self_loops,
            normalize=normalize,
            bias=bias,
            **kwargs,
        )
        
        self.mca_layer = CrossAttentionLayer(
            embed_dim=out_channels,
            kv_dim=mca_kv_dim,
            num_heads=mca_num_heads,
            dropout=mca_dropout,
            scaler_learned=scaler_learned,
            scaler_init=scaler_init
        )
        self.mca_layer.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        graphs_batch: Optional[Tensor] = None,
        extra_kv: Optional[Tensor] = None,
        extra_kv_mask: Optional[Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        This is a modified version of the original Torch Geometric implementation that
        accepts PyG Data objects directly and supports additional key-value features
        that are passed to the message method.
        Args:
            x (Tensor): Node features.
            edge_index (Tensor): Graph connectivity in COO format.
            batch (Optional[Tensor], optional): Batch vector, where each element
                                                indicates the graph a node belongs to.
                                                Defaults to None.
            extra_kv (Optional[Tensor], optional): Additional key-value features to be used in
                                                   message passing.
            extra_kv_mask (Optional[Tensor], optional): Mask for the extra key-value features.
                                                         Defaults to None.
            edge_weight (Optional[torch.Tensor], optional): Edge weights for the graph.
                                                           Defaults to None.
        Returns:
            Tensor: Updated node representations after message passing and linear
                    transformations.
        """
        if x.dim() == 3:
            outputs = []
            for batch_index in range(x.shape[0]):
                batch_x = x[batch_index]
                batch_edge_weight = edge_weight
                if batch_edge_weight is not None and batch_edge_weight.dim() > 1:
                    batch_edge_weight = batch_edge_weight[batch_index]
                batch_extra_kv = extra_kv[batch_index:batch_index + 1] if extra_kv is not None else None
                batch_extra_kv_mask = extra_kv_mask[batch_index:batch_index + 1] if extra_kv_mask is not None else None
                batch_graphs = graphs_batch
                if batch_graphs is None:
                    batch_graphs = torch.zeros(batch_x.size(0), dtype=torch.long, device=batch_x.device)

                self._extra_kv = batch_extra_kv
                self._extra_kv_mask = batch_extra_kv_mask
                self._x_j_batch = batch_graphs[edge_index[1]]

                outputs.append(
                    super(MCAGCNConv, self).forward(
                        x=batch_x,
                        edge_index=edge_index,
                        edge_weight=batch_edge_weight,
                    )
                )

                self._extra_kv = None
                self._extra_kv_mask = None
                self._x_j_batch = None

            return torch.stack(outputs, dim=0)

        if graphs_batch is None:
            graphs_batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Store extra key-value features and batch info for use in message function
        self._extra_kv = extra_kv
        self._extra_kv_mask = extra_kv_mask
        self._x_j_batch = graphs_batch[edge_index[1]]

        # Call parent GCNConv forward method (this will call our overridden message method)
        result = super(MCAGCNConv, self).forward(x=x, edge_index=edge_index, edge_weight=edge_weight)

        # Clean up stored variables
        self._extra_kv = None
        self._extra_kv_mask = None
        self._x_j_batch = None

        return result

    def message(self, x_j: Tensor, edge_weight: Optional[Tensor]) -> Tensor:
        """
        Computes messages for graph convolution with multi-head cross-attention.
        This method extends the standard GCN message passing by incorporating
        cross-attention with extra key-value embeddings.
        Args:
            x_j (Tensor): Node features of neighboring nodes.
            edge_weight (Optional[Tensor]): Edge weights for the graph.
        Returns:
            Tensor: Enhanced node messages after applying cross-attention with extra features.
        """
        
        # Standard GCN message passing
        x_j = super(MCAGCNConv, self).message(x_j, edge_weight)

        # Cross-Attention with extra key-value embeddings
        if self._extra_kv is not None:
            x_j = self.mca_layer(x_j, self._x_j_batch, self._extra_kv, self._extra_kv_mask)

        return x_j


class MCAGINConv(pygnn.conv.GINConv):
    """
    Multi-Channel Attention Graph Isomorphism Network Convolution Layer.
    This layer extends the standard GIN convolution by incorporating a cross-attention mechanism
    that allows passed messages to attend to additional key-value embeddings.
    """

    # Used to regulate reset_parameters behavior
    __init = False

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mca_kv_dim: int,
        mca_num_heads: int,
        mca_dropout: float = 0.0,
        scaler_learned: bool = True,
        scaler_init: float = 0.1,
        nn: Optional[torch.nn.Module] = None,
        eps: float = 0.0,
        train_eps: bool = False,
        edge_dim: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            in_channels (int): Input feature dimensionality.
            out_channels (int): Output feature dimensionality.
            mca_kv_dim (int): Dimension of the cross-attention key-value features.
            mca_num_heads (int): Number of attention heads for cross-attention.
            mca_dropout (float, optional): Dropout rate for cross-attention. Defaults to 0.0.
            nn (torch.nn.Module, optional): A neural network (MLP) that maps node features. 
                                          If None, a default MLP will be created.
            eps (float, optional): Initial epsilon value for GIN. Defaults to 0.0.
            train_eps (bool, optional): Whether epsilon is trainable. Defaults to False.
            **kwargs: Additional arguments passed to the parent GINConv.
        """

        # If no neural network is provided, create a default MLP
        if nn is None:
            nn = MLP(
                [in_channels, out_channels, out_channels],
                act='relu',
                dropout=0.0,
            )

        super(MCAGINConv, self).__init__(
            nn=nn,
            eps=eps,
            train_eps=train_eps,
            **kwargs,
        )

        # Create cross-attention layer
        self.mca_layer = CrossAttentionLayer(
            embed_dim=in_channels,  # in GIN, messages have in_channels dimension
            kv_dim=mca_kv_dim,
            num_heads=mca_num_heads if in_channels != 1 else 1,  # Ensure valid number of heads
            dropout=mca_dropout,
            scaler_learned=scaler_learned,
            scaler_init=scaler_init
        )

        self.__init = True

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        super(MCAGINConv, self).reset_parameters()

        # Only reset MCA parameters if this is not called from the parent class during __init__
        if self.__init:
            self.mca_layer.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        graphs_batch: Optional[Tensor] = None,
        extra_kv: Optional[Tensor] = None,
        extra_kv_mask: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        This is a modified version of the original Torch Geometric implementation that
        accepts PyG Data objects directly and supports additional key-value features
        that are passed to the message method.
        Args:
            x (Tensor): Node features.
            edge_index (Tensor): Graph connectivity in COO format.
            batch (Optional[Tensor], optional): Batch vector, where each element
                                                indicates the graph a node belongs to.
                                                Defaults to None.
            extra_kv (Optional[Tensor], optional): Additional key-value features to be used in
                                                   message passing.
            extra_kv_mask (Optional[Tensor], optional): Mask for the extra key-value features.
                                                         Defaults to None.
            **kwargs: Additional arguments passed to the parent GINConv.
        Returns:
            Tensor: Updated node representations after message passing and MLP transformation.
        """

        # Store extra key-value features and batch info for use in message function
        self._extra_kv = extra_kv
        self._extra_kv_mask = extra_kv_mask
        self._x_j_batch = graphs_batch[edge_index[0]]

        # Call parent GINConv forward method (this will call our overridden message method)
        result = super(MCAGINConv, self).forward(x, edge_index, **kwargs)

        # Clean up stored variables
        self._extra_kv = None
        self._extra_kv_mask = None
        self._x_j_batch = None

        return result

    def message(self, x_j: Tensor) -> Tensor:
        """
        Computes messages for graph convolution with multi-head cross-attention.
        This method extends the standard GIN message passing by incorporating
        cross-attention with extra key-value embeddings.
        Args:
            x_j (Tensor): Node features of neighboring nodes.
        Returns:
            Tensor: Enhanced node messages after applying cross-attention with extra features.
        """

        # Standard GIN message passing
        x_j = super(MCAGINConv, self).message(x_j)

        # Cross-Attention with extra key-value embeddings
        if self._extra_kv is not None:
            x_j = self.mca_layer(x_j, 
                                 self._x_j_batch, 
                                 self._extra_kv, 
                                 self._extra_kv_mask)

        return x_j


class MCAGINEConv(pygnn.conv.GINEConv):
    """
    Multi-Channel Attention Graph Isomorphism Network with Edge features (GINE) Convolution Layer.
    Mirrors MCAGINConv but extends GINEConv to accept edge attributes in the message passing
    while providing the same cross-attention augmentation.
    """

    __init = False

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mca_kv_dim: int,
        mca_num_heads: int,
        mca_dropout: float = 0.0,
        scaler_learned: bool = True,
        scaler_init: float = 0.1,
        nn: Optional[torch.nn.Module] = None,
        eps: float = 0.0,
        train_eps: bool = False,
        edge_dim: Optional[int] = None,
        **kwargs,
    ):

        if nn is None:
            nn = MLP(
                [in_channels, out_channels, out_channels],
                act='relu',
                dropout=0.0,
            )

        super(MCAGINEConv, self).__init__(
            nn=nn,
            eps=eps,
            train_eps=train_eps,
            edge_dim=edge_dim,
            **kwargs,
        )

        # Create cross-attention layer
        self.mca_layer = CrossAttentionLayer(
            embed_dim=in_channels,  # messages have in_channels dimension
            kv_dim=mca_kv_dim,
            num_heads=mca_num_heads,
            dropout=mca_dropout,
            scaler_learned=scaler_learned,
            scaler_init=scaler_init
        )

        self.__init = True

        self.reset_parameters()

    def reset_parameters(self):
        super(MCAGINEConv, self).reset_parameters()
        if self.__init:
            self.mca_layer.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        graphs_batch: Optional[Tensor] = None,
        extra_kv: Optional[Tensor] = None,
        extra_kv_mask: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        # Store extra key-value features and batch info for use in message function
        self._extra_kv = extra_kv
        self._extra_kv_mask = extra_kv_mask
        # GINE message typically uses edge_index[0] for source nodes similar to GIN
        self._x_j_batch = graphs_batch[edge_index[0]] if graphs_batch is not None else None

        # Call parent GINEConv forward (this will call our overridden message method)
        result = super(MCAGINEConv, self).forward(x, edge_index, edge_attr=edge_attr, **kwargs)

        # Clean up stored variables
        self._extra_kv = None
        self._extra_kv_mask = None
        self._x_j_batch = None

        return result

    def message(self, x_j: Tensor, edge_attr: Optional[Tensor]) -> Tensor:
        # Standard GINE message passing
        x_j = super(MCAGINEConv, self).message(x_j, edge_attr)

        # Cross-Attention with extra key-value embeddings
        if self._extra_kv is not None:
            # If x_j is of shape [num_edges, F], and mca_layer expects [num_messages, F]
            x_j = self.mca_layer(x_j, self._x_j_batch, self._extra_kv, self._extra_kv_mask)

        return x_j


class MCAGPSConv(tnn.Module):
    """
    Multi-Channel Attention GPS (General, Powerful, Scalable) Graph Transformer Wrapper.
    This wrapper class extends the GPSConv layer by using MCAGINConv as the local message
    passing layer and maintaining the same interface as other MCA layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mca_kv_dim: int,
        mca_num_heads: int,
        mca_dropout: float = 0.0,
        scaler_learned: bool = True,
        scaler_init: float = 0.1,
        heads: int = 1,
        dropout: float = 0.0,
        act: str = 'relu',
        act_kwargs: Optional[dict] = None,
        norm: str = 'batch_norm',
        norm_kwargs: Optional[dict] = None,
        attn_type: str = 'multihead',
        attn_kwargs: Optional[dict] = None,
        edge_dim: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            in_channels (int): Input feature dimensionality for the local conv layer.
            out_channels (int): Output feature dimensionality, must match the GPS channels 
                               for residual connections and attention.
            mca_kv_dim (int): Dimension of the cross-attention key-value features.
            mca_num_heads (int): Number of attention heads for cross-attention.
            mca_dropout (float, optional): Dropout rate for cross-attention. Defaults to 0.0.
            heads (int, optional): Number of multi-head-attentions in GPS. Defaults to 1.
            dropout (float, optional): Dropout probability of intermediate embeddings. Defaults to 0.0.
            act (str, optional): The non-linear activation function to use. Defaults to 'relu'.
            act_kwargs (dict, optional): Arguments passed to the activation function. Defaults to None.
            norm (str, optional): The normalization function to use. Defaults to 'batch_norm'.
            norm_kwargs (dict, optional): Arguments passed to the normalization function. Defaults to None.
            attn_type (str, optional): Global attention type, 'multihead' or 'performer'. Defaults to 'multihead'.
            attn_kwargs (dict, optional): Arguments passed to the attention layer. Defaults to None.
            **kwargs: Additional arguments passed to GPSConv.
        """
        super(MCAGPSConv, self).__init__()
        
        # Store the channel dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # GPSConv requires input features to have the same dimension as its 'channels' parameter
        # for residual connections (h + x) to work. If in_channels != out_channels, we need
        # an input projection to transform features to out_channels before GPS processing.
        self.input_proj = None
        if in_channels != out_channels:
            self.input_proj = torch.nn.Linear(in_channels, out_channels)
        
        # Create the inner MCAGINConv layer for local message passing
        # Inside GPS, the conv operates on out_channels -> out_channels since the input
        # has already been projected to out_channels by either input_proj or by having
        # in_channels == out_channels from the start
        self.local_conv = MCAGINConv(
            in_channels=out_channels,  # GPS input is always out_channels
            out_channels=out_channels, # GPS conv output must match GPS channels
            mca_kv_dim=mca_kv_dim,
            mca_num_heads=mca_num_heads,
            mca_dropout=mca_dropout,
            scaler_learned=scaler_learned,
            scaler_init=scaler_init
        )
        
        # Create the GPS layer with the MCA-enhanced local convolution
        # GPS uses out_channels for its internal operations
        self.gps_conv = GPSConv(
            channels=out_channels,
            conv=self.local_conv,
            heads=heads,
            dropout=dropout,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            attn_type=attn_type,
            attn_kwargs=attn_kwargs,
            **kwargs,
        )
        
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.input_proj is not None:
            torch.nn.init.xavier_uniform_(self.input_proj.weight)
            if self.input_proj.bias is not None:
                torch.nn.init.zeros_(self.input_proj.bias)
        self.local_conv.reset_parameters()
        self.gps_conv.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        graphs_batch: Optional[Tensor] = None,
        extra_kv: Optional[Tensor] = None,
        extra_kv_mask: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Forward pass that maintains the same interface as other MCA layers.
        Args:
            x (Tensor): Node features.
            edge_index (Tensor): Graph connectivity in COO format.
            batch (Optional[Tensor], optional): Batch vector, where each element
                                                indicates the graph a node belongs to.
                                                Defaults to None.
            extra_kv (Optional[Tensor], optional): Additional key-value features to be used in
                                                   message passing by the inner MCAGINConv layer.
            extra_kv_mask (Optional[Tensor], optional): Mask for the extra key-value features.
                                                         Defaults to None.
            edge_attr (Optional[Tensor], optional): Optional edge attributes passed through to
                                                     the internal `GPSConv` layer if configured
                                                     to use edge features. Shape is usually
                                                     [num_edges, edge_dim]. Defaults to None.
            **kwargs: Additional arguments passed to GPSConv.
        Returns:
            Tensor: Updated node representations after GPS processing with MCA-enhanced
                    local message passing.
        """

        # Project input to out_channels if needed for GPS residual connections
        # GPS does h + x, so both h (conv output) and x (input) must have same dimensions
        if self.input_proj is not None:
            x = self.input_proj(x)

        # Call the GPS forward method
        result = self.gps_conv(
            x,
            edge_index,
            batch=graphs_batch,  # Original GPSConv batch arg
            graphs_batch=graphs_batch,  # local_conv batch arg
            extra_kv=extra_kv,
            extra_kv_mask=extra_kv_mask,
            edge_attr=edge_attr,
            **kwargs,
        )

        return result
