import ast
import torch
import json
import os
import numpy as np
import os.path as osp
import pandas as pd
import pickle 
from itertools import repeat
from rdkit import Chem
import torch_geometric.transforms as T
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils import from_networkx, degree, to_networkx


bond_type_to_int = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2}


def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""

    centered = config.data.centered  #true
    if hasattr(config.data, "shift"):
        shift = config.data.shift
    else:
        shift = 0.

    if hasattr(config.data, 'norm'):
        atom_norm, bond_norm = config.data.norm

        # 0.5,1
        assert shift == 0.

        def scale_fn(x, node = False):
            if centered:
                x = x * 2. - 1.
            else:
                x = x
            if node:
                x = x * atom_norm  #0.5 * (-1,1)
            else:
                x = x * bond_norm  #1 * (-1,1)
            return x
        return scale_fn
    else:
        if centered:
            # Rescale to [-1, 1]
            return lambda x: x * 2. - 1. + shift
        else:
            assert shift == 0.
            return lambda x: x


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""

    centered = config.data.centered
    if hasattr(config.data, "shift"):
        shift = config.data.shift
    else:
        shift = 0.

    if hasattr(config.data, 'norm'):
        atom_norm, bond_norm = config.data.norm

        assert shift == 0.

        def inverse_scale_fn(x, atom=False):
            if atom:
                x = x / atom_norm
            else:
                x = x / bond_norm
            if centered:
                x = (x + 1.) / 2.
            else:
                x = x
            return x

        return inverse_scale_fn
    else:
        if centered:
            # Rescale [-1, 1] to [0, 1]
            return lambda x: (x + 1. - shift) / 2.
        else:
            assert shift == 0.
            return lambda x: x


def networkx_graphs(dataset):
    return [to_networkx(dataset[i], to_undirected=True, remove_self_loops=True) for i in range(len(dataset))]





class Dataset:
    def __init__(self, data_dir, max_nodes,batch_size):
        self.adj_matrix_dict, self.node_features_dict, self.house_node_dict, self.urban_attr, \
        self.demand_dict, self.popu_price = self.load_data(data_dir)
        self.max_nodes = max_nodes
        self.batch_size = batch_size
        self.train_data_list, self.test_data_list, self.train_ids, self.sample_ids = self.split_and_preprocess_data()
        self.sample_urban_fea, self.ids, self.house_mask, self.edge = self.select_and_pad_features()
 
    def load_data(self, data_dir):
        adj_matrix_dict = pickle.load(open(os.path.join(data_dir, 'exist.pkl'), 'rb'))  # 邻接矩阵的字典
        node_features_dict = pickle.load(open(os.path.join(data_dir, 'node_one_hot.pkl'), 'rb'))  # 节点特征的字典
        house_node_dict = pickle.load(open(os.path.join(data_dir, 'house_counts.pkl'), 'rb')) 
        urban_attr = pickle.load(open(os.path.join(data_dir, 'node_features.pkl'), 'rb')) 
        demand_dict = pickle.load(open(os.path.join(data_dir, 'demand_new.pkl'), 'rb'))
        popu_price = pickle.load(open(os.path.join(data_dir, 'popu_price.pkl'), 'rb'))

        return adj_matrix_dict, node_features_dict, house_node_dict, urban_attr, demand_dict, popu_price
    


    def split_and_preprocess_data(self):
        keys = list(self.adj_matrix_dict.keys())
        train_size = int(0.8 * len(keys))
        train_keys = keys[:train_size]
        test_keys = keys[train_size:]


        # Split the data into train and test sets 
        train_adj_matrix_dict = {key: self.adj_matrix_dict[key] for key in train_keys}
        train_node_features_dict = {key: self.node_features_dict[key] for key in train_keys}
        train_house_node_dict = {key: self.house_node_dict[key] for key in train_keys}
        train_urban_attr = {key: self.urban_attr[key] for key in train_keys}
        train_demand_dict = {key: self.demand_dict[key] for key in train_keys}
        train_popu_price = {key: self.popu_price[key] for key in train_keys}

        test_adj_matrix_dict = {key: self.adj_matrix_dict[key] for key in test_keys}
        test_node_features_dict = {key: self.node_features_dict[key] for key in test_keys}
        test_house_node_dict = {key: self.house_node_dict[key] for key in test_keys}
        test_urban_attr = {key: self.urban_attr[key] for key in test_keys}
        test_demand_dict = {key: self.demand_dict[key] for key in test_keys}
        test_popu_price = {key: self.popu_price[key] for key in test_keys}

        # data preprocessing
        train_data_list = self.preprocess_data(train_adj_matrix_dict, train_node_features_dict, train_house_node_dict, \
                                               train_urban_attr,  train_demand_dict, train_popu_price, self.max_nodes)
     
        test_data_list = self.preprocess_data(test_adj_matrix_dict, test_node_features_dict, test_house_node_dict, \
                                              test_urban_attr, test_demand_dict, test_popu_price, self.max_nodes)
         
        train_ids = list(train_adj_matrix_dict.keys())
        
        sample_ids = list(test_adj_matrix_dict.keys())

        return train_data_list, test_data_list, train_ids, sample_ids


    def preprocess_data(self, adj_matrix_dict, node_features_dict, house_dict, urban_attr,\
                        demand_dict, popu_price, max_nodes= None):

        # conduct paddding
        data_list = []
        for key in adj_matrix_dict.keys():
            adj_matrix = adj_matrix_dict[key]
            node_features = node_features_dict[key]
            house_count = house_dict[key]
            urban_fea = urban_attr[key]
            ids = list(adj_matrix_dict.keys())
            demand = demand_dict[key]
            pp = popu_price[key]

            scaler = MinMaxScaler()
            adj_matrix = scaler.fit_transform(adj_matrix)
            bond_feat = torch.zeros(max_nodes, max_nodes)
            bond_feat[:adj_matrix.shape[0], :adj_matrix.shape[1]] = torch.tensor(adj_matrix)
          
            node_feat = torch.zeros(max_nodes, 14) 
            node_feat[:node_features.shape[0], :node_features.shape[1]] = torch.FloatTensor(node_features)
         
            node_mask = torch.cat([torch.ones(node_features.shape[0]), torch.zeros(max_nodes - node_features.shape[0])])
            house_node_mask  = torch.cat([torch.ones(house_count), torch.zeros(max_nodes - house_count)])
    
            padded_urban_fea = torch.zeros(max_nodes, 768)
            padded_urban_fea[:urban_fea.shape[0], :urban_fea.shape[1]] = torch.FloatTensor(urban_fea)
            edge_index = torch.nonzero(bond_feat).t().contiguous()
            # conduct Data
            data = Data(
                node_feat =node_feat.unsqueeze(0), 
                node_mask = node_mask.unsqueeze(0),  # Add batch dimension
                bond_feat = bond_feat.unsqueeze(0),  # Add batch dimension
                house_node_mask = house_node_mask.unsqueeze(0),
                urban_feat = padded_urban_fea.unsqueeze(0),
                demand = torch.FloatTensor(demand).unsqueeze(0),
                pp = torch.FloatTensor(pp).unsqueeze(0),
                grid_ids = torch.tensor(key).unsqueeze(0) 
                )
            
            data_list.append(data)
           

        return data_list
    
    def select_and_pad_features(self, feature_dim=768):
        """Extract features from the data dictionary and pad or truncate them to the size of max_nodes."""
        padded_features, node_list, adjs = [],[],[]
        selected_ids = np.random.choice(self.sample_ids, self.batch_size, replace=False)

        for id in selected_ids:
            padded_urban_fea = torch.zeros(self.max_nodes, feature_dim)
            padded_urban_fea[:self.urban_attr[id].shape[0], :self.urban_attr[id].shape[1]] = torch.tensor(self.urban_attr[id])
           
            padded_features.append(padded_urban_fea)
            node = self.urban_attr[id].shape[0]
            node_list.append(node)
            adj = torch.zeros(self.max_nodes, self.max_nodes)
            adj[:self.adj_matrix_dict[id].shape[0], :self.adj_matrix_dict[id].shape[1]] = torch.tensor(self.adj_matrix_dict[id])
            adjs.append(adj)


        house_node = [self.house_node_dict[id] for id in selected_ids]
        c = torch.stack(adjs).unsqueeze(1) 
        return torch.stack(padded_features), selected_ids, house_node, torch.stack(adjs).unsqueeze(1)

def get_nodes(test_ids, nodes):
    node_list = []
    for i in test_ids:
        node_list.append(nodes[i])
    return node_list


