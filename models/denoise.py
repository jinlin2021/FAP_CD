import torch.nn as nn
import torch
import functools
from torch_geometric.utils import dense_to_sparse
import gc
from . import utils, layers
from .hybrid_layer import HybridBlock
import math
import torch.nn.functional as F

get_act = layers.get_act
conv1x1 = layers.conv1x1


class topo_CrossAttention(nn.Module):
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
        Y = Y / math.sqrt(Y.size(-1)) 

        attn = F.softmax(Y, dim=1)
    
        weighted_V = attn * V
        weighted_V = weighted_V.sum(dim=1)
        weighted_V = weighted_V.flatten(start_dim=1)

        if len(shape) == 3:
            weighted_V = weighted_V.reshape(shape)
        
        return weighted_V


@utils.register_model(name='AF2CG')
class AF2CG(nn.Module):
   

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.act = act = get_act(config)
        self.nf = nf = config.model.hidden 
        self.num_hybrid_layers = num_hybrid_layers = config.model.num_hybrid_layers
        dropout = config.model.dropout
        self.embedding_type = embedding_type = config.model.embedding_type.lower()
        self.conditional = conditional = config.model.conditional 
        self.edge_th = config.model.edge_th
        self.rw_depth = rw_depth = config.model.rw_depth

        modules = []
        # timestep/noise_level embedding
        if embedding_type == 'positional':
            embed_dim = nf
        else:
            raise ValueError(f'embedding type {embedding_type} unknown.')

        if conditional:
            modules.append(nn.Linear(embed_dim, nf * 2))
            modules.append(nn.Linear(nf * 2, nf))

        atom_ch = config.data.atom_channels # 14
        bond_ch = config.data.bond_channels # 1
        temb_dim = nf 

        # project edge features
        assert bond_ch == 1 
        bond_se_ch = int(nf * 0.5)  
        modules.append(conv1x1(1, bond_se_ch))
        modules.append(conv1x1(rw_depth + 1, bond_se_ch))
        modules.append(nn.Linear(bond_se_ch*2, nf)) 

        # project node features
        atom_se_ch = int(nf * 0.2)  #  
        atom_type_ch = nf - 2 * atom_se_ch  
        modules.append(nn.Linear(bond_ch, atom_se_ch)) 
        modules.append(nn.Linear(atom_ch, atom_type_ch)) 
        modules.append(nn.Linear(rw_depth, atom_se_ch))  
        modules.append(nn.Linear(atom_type_ch + 2 * atom_se_ch, nf))  
        self.x_ch = nf

        # Residual Hybrid Layer network
        cat_dim = (nf * 2) // num_hybrid_layers  #  num_gnn_layers 3 层
        for _ in range(num_hybrid_layers):
            modules.append(HybridBlock(self.config, nf, config.model.graph_layer, "FullTrans_1", config.model.heads,
                                         temb_dim=temb_dim, act=act, dropout=dropout, attn_dropout=dropout))
            modules.append(nn.Linear(nf, cat_dim)) # (nf, nf*2 //3)
            modules.append(nn.Linear(nf, cat_dim))

        # atom output
        #(48*2//3 * 3 +nf - 2 * atom_se_ch, 256)
        modules.append(nn.Linear(cat_dim * num_hybrid_layers + atom_type_ch, nf))
        modules.append(nn.Linear(nf, nf // 2))
        modules.append(nn.Linear(nf // 2, atom_ch))

        # bond structure output
        modules.append(conv1x1(cat_dim * num_hybrid_layers + bond_se_ch, nf))
        modules.append(conv1x1(nf, nf // 2))
        modules.append(conv1x1(nf // 2, 1))

        self.all_modules = nn.ModuleList(modules)

        self.linear_urban_emb = nn.Linear(14, nf)
        self.cross_attn_x1 = topo_CrossAttention(xdim = config.model.hidden, n_head = 2) 
        


    def forward(self, x, time_cond, *args, **kwargs):
     
        atom_feat, bond_feat = x   # [B,maxnodes,14]  [B,1,maxnodes,maxnodes]
        atom_mask = kwargs['atom_mask']   # [B,maxnodes]
        bond_mask = kwargs['bond_mask']  #[B,1,maxnodes,maxnodes]
        urban_feat = kwargs['urban_attr']  #[B,maxnodes,768] 
        edge_exist = bond_feat
      
      
        modules = self.all_modules
        m_idx = 0

        if self.embedding_type == 'positional':
            # Sinusoidal positional embeddings.
            timesteps = time_cond
            temb = layers.get_timestep_embedding(timesteps, self.nf)  # [B,48]

        else:
            raise ValueError(f'embedding type {self.embedding_type} unknown.')

        if self.conditional:
            # True
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))  #[B,48]
            m_idx += 1
        else:
            temb = None
      
        if not self.config.data.centered:
            # rescale the input data to [-1, 1]  
            atom_feat = atom_feat * 2. - 1.
            bond_feat = bond_feat * 2. - 1.

        # discretize dense adj
        with torch.no_grad():
            adj = edge_exist.squeeze(1).clone()  # [B, N, N]
            adj[adj >= 0.] = 1.
            adj[adj < 0.] = 0.
            adj = adj * bond_mask.squeeze(1)  # [B, N, N]

        # extract RWSE and Shortest-Path Distance
        rw_landing, spd_onehot = utils.get_rw_feat(self.rw_depth, adj)

        # construct edge feature [B, N, N, F]
        adj_mask = bond_mask.permute(0, 2, 3, 1) #[B, N, N, 1]

        dense_exist = modules[m_idx](edge_exist).permute(0, 2, 3, 1) * adj_mask # [B, N, N, 24]
        # print(modules[m_idx])
        m_idx += 1
        dense_spd = modules[m_idx](spd_onehot).permute(0, 2, 3, 1) * adj_mask # [B, N, N, 24]
        m_idx += 1
        # print(modules[m_idx])
        st = torch.cat([dense_exist, dense_spd], dim=-1) #torch.Size([B, 38, 38, NF])
        dense_edge = modules[m_idx](st) * adj_mask  # [B, N, N, 48]

       
        m_idx += 1

        # Use Degree as atom feature
        atom_degree = torch.sum(bond_feat, dim=-1).permute(0, 2, 1)  # [B, N, 9]
        atom_degree = modules[m_idx](atom_degree)  # [B, N, 9]
        m_idx += 1
        atom_cate = modules[m_idx](atom_feat) # [B, N, 30]
        m_idx += 1
        x_rwl = modules[m_idx](rw_landing) # [B, N, 9]
        m_idx += 1
        x_atom = modules[m_idx](torch.cat([atom_degree, atom_cate, x_rwl], dim=-1)) # [B, N, 48]
        m_idx += 1


        dense_index = adj.nonzero(as_tuple=True)
        edge_index, _ = dense_to_sparse(adj)
        h_dense_edge = dense_edge

        
        atom_hids = []
        bond_hids = []
        
        urban_feat = self.linear_urban_emb(urban_feat) #[B*N,48]
        for _ in range(self.num_hybrid_layers):
            
            x_atom, h_dense_edge = modules[m_idx](x_atom, edge_index, h_dense_edge, dense_index,
                                                  atom_mask, adj_mask, temb, urban_feat)
          
            m_idx += 1
            atom_hids.append(modules[m_idx](x_atom.reshape(x_atom.shape)))
            m_idx += 1
            bond_hids.append(modules[m_idx](h_dense_edge))
            m_idx += 1
        

        atom_hids = torch.cat(atom_hids, dim=-1) # [B, 400, 510] # 
        bond_hids = torch.cat(bond_hids, dim=-1)  #[B, N, N, 510]


        # Output  
        atom_score = self.act(modules[m_idx](torch.cat([atom_cate, atom_hids], dim=-1))) \
                     * atom_mask.unsqueeze(-1)  ## [B, 400, 256]
        m_idx += 1
        atom_score = self.act(modules[m_idx](atom_score)) ## [B, 400, 128]
        m_idx += 1
        atom_score = modules[m_idx](atom_score) #  [B, 400, 14]
        m_idx += 1

        exist_score = self.act(modules[m_idx](torch.cat([dense_exist, bond_hids], dim=-1).permute(0, 3, 1, 2))) \
                      * bond_mask  # torch.Size([B, 256, N, N])
        m_idx += 1
        exist_score = self.act(modules[m_idx](exist_score))  #[B, 128, N, N]
        m_idx += 1
        exist_score = modules[m_idx](exist_score) #torch.Size([B, 1, 400, 400])
        m_idx += 1


        bond_score = (exist_score + exist_score.transpose(2, 3)) / 2.
        assert m_idx == len(modules)
  
        atom_score = atom_score * atom_mask.unsqueeze(-1)  # torch.Size([4, 400, 14])
        bond_score = bond_score * bond_mask  #torch.Size([4, 1, 400, 400])

        return atom_score, bond_score


    def get_type(self, atom_score, bond_score,**kwargs):
        atom_score[:,:,6] ==  -1e9
        atom_feature = atom_score.clone()
        atom_feature = atom_feature.argmax(dim=-1)
        return atom_feature

