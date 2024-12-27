import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn

from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.utils import dense_to_sparse

from .graph_transformer import GraphTransformerLayer
import gc



class CrossAttention(nn.Module):
    def __init__(self, xdim, n_head):
        super().__init__()

        
        self.n_head = n_head
        self.df = int(xdim / self.n_head)

        self.q = nn.Linear(xdim, xdim)
        self.k = nn.Linear(xdim, xdim)
        self.v = nn.Linear(xdim, xdim)

    def forward(self, x, c):
        shape = x.shape
        if len(shape) == 3:
            x = x.reshape([-1, x.shape[-1]])
            c = c.reshape([-1, c.shape[-1]])

        Q, K, V = self.q(x), self.k(c), self.v(c)

        Q = Q.reshape((Q.size(0), self.n_head, self.df)) # self.n_head * self.df  = xdim
        K = K.reshape((K.size(0), self.n_head, self.df))
        V = V.reshape((V.size(0), self.n_head, self.df))
        

        Q = Q.unsqueeze(1)                             # (n, 1, n_head, df)
        K = K.unsqueeze(0)                             # (1, n, n head, df)
        V = V.unsqueeze(0)                             # (1, n, n_head, df)

        Y = Q * K
        Y = Y / math.sqrt(Y.size(-1)) # 3603

        attn = F.softmax(Y, dim=1)
        # 注意力权重的计算是基于查询Q和键K之间的相似度，如果键和查询的维度较小，那么这个计算会更快。

        weighted_V = attn * V
        weighted_V = weighted_V.sum(dim=1)
        weighted_V = weighted_V.flatten(start_dim=1)

        if len(shape) == 3:
            weighted_V = weighted_V.reshape(shape)
        
        return weighted_V



class HybridBlock(nn.Module):
    """Local MPNN + graph transformer layer. """

    def __init__(self, config, dim_h,
                 local_gnn_type, global_model_type, num_heads,
                 temb_dim=None, act=None, dropout=0.0, attn_dropout=0.0):
        super().__init__()

        self.dim_h = dim_h
        self.config = config
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.local_gnn_type = local_gnn_type
        self.global_model_type = global_model_type
        if act is None:
            self.act = nn.ReLU()
        else:
            self.act = act

        # time embedding
        if temb_dim is not None:
            self.t_node = nn.Linear(temb_dim, dim_h)
            self.t_edge = nn.Linear(temb_dim, dim_h)
            self.t_node_feat = nn.Linear(temb_dim, dim_h)

        # local message-passing model
        if local_gnn_type == 'None':
            self.local_model = None
        elif local_gnn_type == 'GCN':
            # gine 
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h), nn.ReLU(), Linear_pyg(dim_h, dim_h))
            # gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h //2), nn.ReLU(), Linear_pyg(dim_h //2, dim_h))
            #  Linear_pyg 是PyTorch Geometric 针对图神经网络设计的特化线性层
            self.local_model = pygnn.GINEConv(gin_nn)

        # Global attention transformer-style model.
        if global_model_type == 'None':
            self.self_attn = None
        elif global_model_type == 'FullTrans_1': 
            #使用 FullTrans_1 
            # self.self_attn = EdgeGateTransLayer(dim_h, dim_h // num_heads, num_heads, edge_dim=dim_h)
            self.transformerLayer = GraphTransformerLayer(self.config)
 

        # Normalization for MPNN and Self-Attention representations.
        self.norm1_local = nn.GroupNorm(num_groups=min(dim_h // 4, 32), num_channels=dim_h, eps=1e-6)
        self.norm2 = nn.GroupNorm(num_groups=min(dim_h // 4, 32), num_channels=dim_h, eps=1e-6)

        self.dropout = nn.Dropout(dropout)

        # Feed Forward block -> node.
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        self.norm2_node = nn.GroupNorm(num_groups=min(dim_h // 4, 32), num_channels=dim_h, eps=1e-6)

        # Feed Forward block -> edge.
        self.ff_linear3 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear4 = nn.Linear(dim_h * 2, dim_h)
        self.norm2_edge = nn.GroupNorm(num_groups=min(dim_h // 4, 32), num_channels=dim_h, eps=1e-6)

        self.cross_attn_x1 = CrossAttention(xdim = config.model.hidden, n_head = 2)  #多头注意力注意力机制是 2


    def _ff_block_node(self, x):
        """Feed Forward block.
        """
        x = self.dropout(self.act(self.ff_linear1(x)))
        return self.dropout(self.ff_linear2(x))

    def _ff_block_edge(self, x):
        """Feed Forward block.
        """


        x = self.dropout(self.act(self.ff_linear3(x)))
        return self.dropout(self.ff_linear4(x))

    def forward(self, x, edge_index, dense_edge, dense_index, node_mask, adj_mask, cond_temb=None, urban_feat= None):
        """
        Args:
            x: node feature [B*N, dim_h]
            edge_index: [2, edge_length] 
            dense_edge: edge features in dense form [B, N, N, dim_h] 
            dense_index: indices for valid edges [B, N, N, 1] 
            node_mask: [B, N]
            adj_mask: [B, N, N, 1]
            temb: time conditional embedding [B, temb_dim]
        Returns:
            node
            edge
        """

        B, N, _, _ = dense_edge.shape # B, N, N, 48
        h_in1 = x.reshape(-1,self.dim_h)  #[B*N,48]
        h_in2 = dense_edge #[B, N, N, 256]

        if cond_temb is not None:
            cond_temb = (urban_feat + self.t_node_feat(self.act(cond_temb))[:, None, :]) #[b,n,f]
            node = (x + self.t_node(self.act(cond_temb))).reshape(-1, self.dim_h)* node_mask.reshape(-1, 1) #[B*N,48]
            h_edge = (dense_edge + self.t_edge(self.act(cond_temb))[:, None, :, :]) * adj_mask
  

        # MPNN_out_list = []
        # Local MPNN 
        if self.local_model is not None:
            edge_attr = h_edge[dense_index] 
            h_local = self.local_model(node, edge_index, edge_attr) * node_mask.reshape(-1, 1)
            h_local = h_in1 + self.dropout(h_local)
            h_MPNN = self.norm1_local(h_local) # group normalization
            # MPNN_out_list.append(h_local)
            # h_MPNN = sum(MPNN_out_list) * node_mask.reshape(-1, 1)
            h_dense_MPNN = h_MPNN.reshape(B, N, -1)  # [B, 400, hiddden]
     
            h_dense_MPNN = h_dense_MPNN.unsqueeze(1) + h_dense_MPNN.unsqueeze(2) 
            h_MPNN = h_MPNN + self._ff_block_node(h_MPNN) 
            h_MPNN = self.norm2_node(h_MPNN) * node_mask.reshape(-1, 1) 
            h_dense_MPNN = h_in2 + self._ff_block_edge(h_dense_MPNN)
            h_dense_MPNN = self.norm2_edge(h_dense_MPNN.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) * adj_mask # [B, 400, 400, 256]


            

        # Multi-head attention with graph transformer 
        # Trans_out_list = []
        if self.transformerLayer is not None:
            if 'FullTrans' in self.global_model_type:
             
                h_attn, edge_attn = self.transformerLayer(node, h_edge) 
                h_attn = h_attn.reshape(-1, h_attn.shape[-1])
                h_attn = h_attn * node_mask.reshape(-1, 1)
           
                h_attn = h_in1 + self.dropout(h_attn)
                h_attn = self.norm2(h_attn)
                # Trans_out_list.append(h_attn)
                # h_attn = sum(Trans_out_list) * node_mask.reshape(-1, 1)
                h_attn = self.norm2_node(h_attn) * node_mask.reshape(-1, 1) #[1600,256]

                h_trans = h_attn.reshape(B, N, -1)  # [B, 400, 256]
                h_trans = h_trans.unsqueeze(1) + h_trans.unsqueeze(2)
                h_trans = self.norm2_edge(h_trans.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) * adj_mask  
        
        node =  h_MPNN + h_attn
        node = node.reshape(-1, N, h_in1.shape[-1])
        edge = h_dense_MPNN + h_trans
       
        return node, edge

