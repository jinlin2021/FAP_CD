
import os

import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from tqdm import tqdm




class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        xdim = config["GTN_x_hiddim"]
        self.n_head = config.model.heads
        self.df = int(xdim / self.n_head)

        self.q = nn.Linear(xdim, xdim)
        self.k = nn.Linear(xdim, xdim)
        self.v = nn.Linear(xdim, xdim)

    def forward(self, x, c):
        shape = x.shape
        if len(shape) == 3:
            x = x.reshape([-1, x.shape[-1]])
            c = c.reshape([-1, c.shape[-1]])

        Q, K, V = self.q(x), self.k(c), self.v(c) # q 查询，找到 对与 k 的重要性，就是 attention，乘以 V，也就是更新后的表示 C

        Q = Q.reshape((Q.size(0), self.n_head, self.df))
        K = K.reshape((K.size(0), self.n_head, self.df))
        V = V.reshape((V.size(0), self.n_head, self.df))
        

        Q = Q.unsqueeze(1)                             # (n, 1, n_head, df)
        K = K.unsqueeze(0)                             # (1, n, n head, df)
        V = V.unsqueeze(0)                             # (1, n, n_head, df)

        Y = Q * K
        Y = Y / math.sqrt(Y.size(-1)) # 3603

        attn = F.softmax(Y, dim=1)

        weighted_V = attn * V
        weighted_V = weighted_V.sum(dim=1)
        weighted_V = weighted_V.flatten(start_dim=1)

        if len(shape) == 3:
            weighted_V = weighted_V.reshape(shape)
        
        return weighted_V
    

class NodeEdgeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        xdim = config["GTN_x_hiddim"]  #nf 256
        edim = config["GTN_e_hiddim"]   # 128
        self.n_head = config.model.heads  # 8
        self.df = int(xdim / self.n_head)
       
        # Attention
        self.q = nn.Linear(xdim, xdim)
        self.k = nn.Linear(xdim, xdim)
        self.v = nn.Linear(xdim, xdim)

        # FiLM E to X
        self.e_add = nn.Linear(edim, xdim)
        self.e_mul = nn.Linear(edim, xdim)

        # Output layers
        self.x_out = nn.Linear(xdim, xdim)
        self.e_out = nn.Linear(xdim, edim)

    def forward(self, x, e):
        # Map X to keys and queries
        
        Q = self.q(x)
        K = self.k(x) 
        
        # Reshape to (n, n_head, df) with dx = n_head * df
        Q = Q.reshape((Q.size(0), Q.size(1), self.n_head, self.df))  # (b, n,nhead, xdim / n_head) 
        K = K.reshape((K.size(0), Q.size(1), self.n_head, self.df))
        
        Q = Q.unsqueeze(2)                             # (b,n, 1, n_head, df)
        K = K.unsqueeze(1)                             # (b,1, n,n_head, df)

        # Compute unnormalized attentions.
        Y = Q * K  # [B,N,N,nhead,32]
        Y = Y / math.sqrt(Y.size(-1)) 
        
        E1 = self.e_mul(e)                           
        E1 = E1.reshape((e.size(0), e.size(1), e.size(2), self.n_head, self.df)) # （B,N,N, head, xdim / n_head）
        
        E2 = self.e_add(e)
        E2 = E2.reshape((e.size(0), e.size(1), e.size(2), self.n_head, self.df)) # 4085
        
        # Incorporate edge features to the self attention scores.
        Y = Y * (E1 + 1) + E2    
        # Output E
        newE = Y.flatten(start_dim=3) 
        newE = self.e_out(newE) # 5531 
        
        # Compute attentions. attn is still (n, n, n_head, df)
        attn = F.softmax(Y, dim=1) # * adj.unsqueeze(-1).unsqueeze(-1) # 6495
        
        # Map X to values
        V = self.v(x)                        # n, dx
        V = V.reshape((V.size(0), V.size(1),self.n_head, self.df))        
        V = V.unsqueeze(1)                  
        # Compute weighted values
        weighted_V = attn * V                
        weighted_V = weighted_V.sum(dim=1)   
        
        # Send output to input dim
        weighted_V = weighted_V.flatten(start_dim=2)  

        # Output X
        newX = self.x_out(weighted_V) 
        
        return newX, newE
    


class GraphTransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Set dimensions
        xdim, edim = config.GTN_x_hiddim, config.GTN_e_hiddim
        dim_ffX, dim_ffE = config.GTN_dim_ffX, config.GTN_dim_ffE

        # Initialize node and edge processing modules
        self.self_attn = NodeEdgeBlock(config)
        self.dropout  =config.model.dropout

        # Initialize linear, normalization, and dropout layers
        self.linX1, self.linX2, self.normX1, self.normX2, self.dropoutX = self.create_ff_layers(xdim, dim_ffX, self.dropout)
        self.linE1, self.linE2, self.normE1, self.normE2, self.dropoutE = self.create_ff_layers(edim, dim_ffE, self.dropout)

    def create_ff_layers(self, input_dim, ff_dim, dropout_rate):
        lin1 = nn.Linear(input_dim, ff_dim)
        lin2 = nn.Linear(ff_dim, input_dim)
        norm1 = nn.LayerNorm(input_dim)
        norm2 = nn.LayerNorm(input_dim)
        dropout = nn.Dropout(dropout_rate)
        
        return lin1, lin2, norm1, norm2, dropout


    def forward_ff_block(self, input, new, lin1, lin2, norm, dropout):
        # Forward feed layer processing
        res_input = input
        input = norm(res_input + dropout(new))
        input = F.relu(lin1(input), inplace=True)
        input = lin2(dropout(input))
        input = norm(res_input + dropout(input))
        return input


    def forward(self, x, e):
        # Self-attention layer processing
        x= x.reshape([-1, e.size(1),x.shape[-1]])
        
        newX, newE = self.self_attn(x, e) # 输出 256 # 【batchsize，400,64】
        
        # Process node features
        # 输出 256
        x = self.forward_ff_block(x, newX, self.linX1, self.linX2, self.normX2, self.dropoutX)
        
        # Process edge features
        e = self.forward_ff_block(e, newE, self.linE1, self.linE2, self.normE2, self.dropoutE)
        
        return x, e


