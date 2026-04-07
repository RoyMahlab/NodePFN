from functools import partial

from typing import Tuple
from torch import nn
import torch
from torch.nn.modules.transformer import _get_activation_fn, Module, Tensor, Optional, MultiheadAttention, Linear, Dropout, LayerNorm
from torch.utils.checkpoint import checkpoint
import torch_geometric.nn as pygnn
from torch_geometric.data import Data
from torch_geometric.nn import Linear as Linear_pyg
import torch.nn.functional as F
from nodepfn.ginat.message_cagcn_layer import MCAMPNN
from nodepfn.ginat.layers import resolve_target_backbone

class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.
        local_gnn_type: Type of local GNN to use ('None', 'GCN', 'GIN', 'GraphSAGE', 'GAT').
        use_gps_style: Whether to use GPS-style combination of local MPNN and global attention.

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=False, pre_norm=False,
                 device=None, dtype=None, recompute_attn=False, local_gnn_type='GCN', 
                 use_gps_style=True, conv_type='gcn') -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.use_gps_style = use_gps_style
        self.local_gnn_type = local_gnn_type
        
        # Self-attention (global transformer)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        
        local_gnn_type = "GCN"
        # Local GNN setup (GPS-style)
        if use_gps_style and local_gnn_type != 'None':
            self.local_gnn_with_edge_attr = True
            if local_gnn_type == "GCN":
                self.local_gnn_with_edge_attr = False
                # self.local_model = pygnn.GCNConv(d_model, d_model)
                self.local_model = self.local_model = MCAMPNN(
                    conv_cls=resolve_target_backbone(conv_type, False),
                    node_features_channels=d_model,
                    extra_features_channels=4096,
                    out_channels=d_model,
                    hidden_channels=512,
                    num_layers=1,
                    attn_heads=4
                )
            elif local_gnn_type == 'GIN':
                self.local_gnn_with_edge_attr = False
                gin_nn = nn.Sequential(Linear_pyg(d_model, d_model),
                                       nn.ReLU(),
                                       Linear_pyg(d_model, d_model))
                self.local_model = pygnn.GINConv(gin_nn)
            elif local_gnn_type == 'GraphSAGE':
                self.local_model = pygnn.SAGEConv(d_model, d_model)
            elif local_gnn_type == 'GAT':
                self.local_model = pygnn.GATConv(in_channels=d_model,
                                                 out_channels=d_model // nhead,
                                                 heads=nhead,
                                                 edge_dim=d_model)
            else:
                raise ValueError(f"Unsupported local GNN model: {local_gnn_type}")
        else:
            self.local_model = None
            
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        # Normalization layers
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        
        # GPS-style: separate norms for local and global
        if use_gps_style:
            self.norm1_local = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm1_global = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs) 
        
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        if use_gps_style:
            self.dropout_local = Dropout(dropout)
            self.dropout_global = Dropout(dropout)
        
        self.pre_norm = pre_norm
        self.recompute_attn = recompute_attn

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, 
                edge_index: Optional[Tensor] = None, extra_features: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            edge_index: edge indices for local GNN (optional, required if using GPS-style).

        Shape:
            see the docs in Transformer class.
        """
        h = src
        h_in1 = h  # for first residual connection
        if self.use_gps_style:
            # GPS-style: combine local MPNN and PFN attention
            h_out_list = []
            
            # Local MPNN processing
            if self.local_model is not None and len(edge_index) != 0:
                if self.pre_norm:
                    h_local_input = self.norm1_local(h)
                else:
                    h_local_input = h
                    
                if isinstance(self.local_model, MCAMPNN):
                    if extra_features is None:
                        h_local = None
                    else:
                        prompt_features, prompt_mask = extra_features
                        prompt_features = prompt_features.to(h_local_input.device)
                        prompt_mask = prompt_mask.to(h_local_input.device)

                        if h_local_input.dim() == 3:
                            batch_outputs = []
                            for b in range(h_local_input.shape[1]):
                                graph = Data(x=h_local_input[:, b, :], edge_index=edge_index)
                                batch_prompt = (prompt_features[b:b+1], prompt_mask[b:b+1])
                                batch_outputs.append(self.local_model(graph, extra_features=batch_prompt))
                            h_local = torch.stack(batch_outputs, dim=1)
                        else:
                            graph = Data(x=h_local_input, edge_index=edge_index)
                            h_local = self.local_model(graph, extra_features=(prompt_features, prompt_mask))
                else:
                    h_local = self.local_model(h_local_input.transpose(0,1), edge_index).transpose(0,1)

                if h_local is None:
                    h_local = torch.zeros_like(h)
                
                h_local = self.dropout_local(h_local)
                h_local = h_in1 + h_local  # Residual connection
                
                if not self.pre_norm:
                    h_local = self.norm1_local(h_local)
                    
                h_out_list.append(h_local)
            
            # PFN attention processing
            if self.pre_norm:
                h_pfn_input = self.norm1_global(h)
            else:
                h_pfn_input = h
                
            h_pfn = self._process_global_attention(h_pfn_input, src_mask, src_key_padding_mask)
            h_pfn = self.dropout_global(h_pfn)
            h_pfn = h_in1 + h_pfn  # Residual connection
            
            if not self.pre_norm:
                h_pfn = self.norm1_global(h_pfn)
                
            h_out_list.append(h_pfn)
            
            # Combine local and global outputs
            if len(h_out_list) > 1:
                h = sum(h_out_list)  # sum the outputs
            else:
                h = h_out_list[0]
        else:
            # Original transformer processing
            if self.pre_norm:
                src_ = self.norm1(src)
            else:
                src_ = src
                
            src2 = self._process_global_attention(src_, src_mask, src_key_padding_mask)
            src = src + self.dropout1(src2)
            
            if not self.pre_norm:
                src = self.norm1(src)
            h = src

        # Feed Forward block
        if self.pre_norm:
            h_ = self.norm2(h)
        else:
            h_ = h
            
        h2 = self.linear2(self.dropout(self.activation(self.linear1(h_))))
        h = h + self.dropout2(h2)

        if not self.pre_norm:
            h = self.norm2(h)
            
        return h
    
    def _process_global_attention(self, src_: Tensor, src_mask: Optional[Tensor] = None, 
                                  src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """Process attention with the original PFN logic."""
        if isinstance(src_mask, tuple):
            # pfn attention setup
            assert not self.self_attn.batch_first
            assert src_key_padding_mask is None

            global_src_mask, trainset_src_mask, valset_src_mask = src_mask

            num_global_tokens = global_src_mask.shape[0]
            num_train_tokens = trainset_src_mask.shape[0]

            global_tokens_src = src_[:num_global_tokens]
            train_tokens_src = src_[num_global_tokens:num_global_tokens+num_train_tokens]
            global_and_train_tokens_src = src_[:num_global_tokens+num_train_tokens]
            eval_tokens_src = src_[num_global_tokens+num_train_tokens:]

            attn = partial(checkpoint, self.self_attn) if self.recompute_attn else self.self_attn

            global_tokens_src2 = attn(global_tokens_src, global_and_train_tokens_src, global_and_train_tokens_src, None, True, global_src_mask)[0]
            train_tokens_src2 = attn(train_tokens_src, global_tokens_src, global_tokens_src, None, True, trainset_src_mask)[0]
            eval_tokens_src2 = attn(eval_tokens_src, src_, src_,
                                    None, True, valset_src_mask)[0]

            src2 = torch.cat([global_tokens_src2, train_tokens_src2, eval_tokens_src2], dim=0)

        elif isinstance(src_mask, int):
            assert src_key_padding_mask is None
            single_eval_position = src_mask
            src_left = self.self_attn(src_[:single_eval_position], src_[:single_eval_position], src_[:single_eval_position])[0]
            src_right = self.self_attn(src_[single_eval_position:], src_[:single_eval_position], src_[:single_eval_position])[0]
            src2 = torch.cat([src_left, src_right], dim=0)
        else:
            if self.recompute_attn:
                src2 = checkpoint(self.self_attn, src_, src_, src_, src_key_padding_mask, True, src_mask)[0]
            else:
                src2 = self.self_attn(src_, src_, src_, attn_mask=src_mask,
                                      key_padding_mask=src_key_padding_mask)[0]
        return src2
