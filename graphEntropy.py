import torch
import torch.nn as nn
import torch.nn.functional as F
import math



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
        Y = Y / math.sqrt(Y.size(-1)) 

        attn = F.softmax(Y, dim=1)
       

        weighted_V = attn * V
        weighted_V = weighted_V.sum(dim=1)
        weighted_V = weighted_V.flatten(start_dim=1)

        if len(shape) == 3:
            weighted_V = weighted_V.reshape(shape)
        
        return weighted_V
    

class ConditionNetwork(nn.Module):

    def __init__(self, config, dropout_rate=0.3):
        super(ConditionNetwork, self).__init__()
        self.dim = config.model.size  
        hids =[13, 768, 2]
        self.l1 = nn.Linear(hids[0] +2 , self.dim)
        self.l2 = nn.Linear(hids[1], self.dim)
        self.l3 = nn.Linear(hids[2], hids[0]+1)
        self.linear = nn.Linear((hids[0]+1) *2 , hids[0]+1)
        n_head = 2
        self.attention = CrossAttention(self.dim, n_head)   
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim ),
            nn.ReLU(),
            nn.Linear(self.dim , self.dim // 2),
            nn.ReLU(),
            nn.Linear(self.dim // 2, hids[0]+1 )
        )
        self.dropout = nn.Dropout(dropout_rate)
      
        self.norm = nn.BatchNorm1d(self.dim // 2) # 
        self.act = nn.ReLU()
        # self.norm = nn.GroupNorm(num_groups=16, num_channels= 128, eps=1e-6)

    def forward(self, demand, attr, pp, node_mask):

        demand = torch.cat([demand, pp], dim=1) 
        demand = self.l1(demand).unsqueeze(1) 
        demand = demand.expand(-1, attr.size()[1], -1)  
        attr = self.l2(attr) 
        attr = self.attention(demand,attr) * attr #[b,n,model.size]
        attr = self.mlp(attr) * node_mask.unsqueeze(-1) 
       
        attr = attr * node_mask.unsqueeze(-1)
        return attr 
    
    def compute_loss_x(self, attr, node_mask):  
        """
        args:  
        attr, 
        node_mask
        return:
        min_entropy in the batch
        """
 
        probabilities = F.softmax(attr, dim=-1) #【B,400,14】
        probabilities = probabilities.masked_fill(node_mask.unsqueeze(-1) == 0, float('0'))
        # print(probabilities[0,0,6])
        probabilities = probabilities.sum(dim=1) #[8,,14]
        probabilities[:,6] = 0
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True) # 归一化[8,14]
        log_prob = torch.log(probabilities + 1e-6)
        entropy = -torch.sum(probabilities * log_prob, dim=1) 
        min_entropy = entropy.min()
        loss = 1 / (1+ min_entropy)
        return loss
    
   
    

def maximize_entropy_min_loss(entropy,house_node_mask):
    """
    args:
    entropy (torch.Tensor): entropy of the predicted distribution in the batch
    house_node_mask (torch.Tensor): mask of the house nodes in the batch
    return:
    torch.Tensor:
    """
   
    entropy_sum = torch.sum(entropy, dim=1)
    # Calculate the sum of non-zero elements in each row of house_node_mask
    non_zero_sums = torch.sum(house_node_mask != 0, dim=1).float()
    entropy_mean = entropy_sum / non_zero_sums
    min_entropy = entropy_mean.min()
    loss = -min_entropy
    return loss

def normalize_tensor(tensor, dim=2):
    """
    args:
    tensor (torch.Tensor): 
    dim (int): 
    return:
    torch.Tensor: 
    """
   
    min_vals = tensor.min(dim=dim, keepdim=True).values
    max_vals = tensor.max(dim=dim, keepdim=True).values

   
    eps = 1e-8
    normalized_tensor = (tensor - min_vals) / (max_vals - min_vals + eps)
    return normalized_tensor

