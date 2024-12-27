import os
import torch
import numpy as np
import random
import logging
import time
from absl import flags
from torch.utils import tensorboard
from torch_geometric.loader import DataLoader, DenseDataLoader
import torch.optim as optim
import pickle
from rdkit import RDLogger, Chem

from models import denoise
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
from datasets import *
import datasets
import sde_lib
from utils import *
import torch_geometric.transforms as T
from graphEntropy import ConditionNetwork


FLAGS = flags.FLAGS
from tqdm import tqdm

import gc

def set_random_seed(config):
    seed = config.seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

          


def train_cond_model(cond_pred_model, dataloader, opt, scaler):
    cond_pred_model.train()
    losses  =[]
    for batch in dataloader:
        opt.zero_grad()
        batch = dense_mol(batch, scaler)
        #输入的数据集
        node_feat, node_mask, bond_feat, bond_mask, house_node_mask, urban_attr,\
            demand, pp, grid_ids = batch
        pred = cond_pred_model(demand, urban_attr,pp, node_mask) # bs*1 预测维度需要是 14 
        loss = cond_pred_model.compute_loss_x(pred, node_mask)
        # loss = cond_pred_model.compute_loss_x1(pred, node_mask, house_node_mask)       
        loss.backward()
        opt.step()
        losses.append(loss.item())
    # print('loss:%.3f'%np.mean(losses))
    return np.mean(losses)

def eval_cond_model(cond_pred_model, dataloader, scaler):
    cond_pred_model.eval()
    with torch.no_grad():
        losses  =[]
        emb ={}
        for batch in dataloader:
            batch = dense_mol(batch, scaler)
            #输入的数据集
            node_feat, node_mask, bond_feat, bond_mask, house_node_mask, urban_attr,\
                demand, pp, grid_ids = batch

            pred = cond_pred_model(demand, urban_attr, pp, node_mask) # bs*1 预测维度需要是 14 
            loss = cond_pred_model.compute_loss_x(pred, node_mask,house_node_mask)
        
            losses.append(loss.item())
            pred = pred.cpu().numpy()
            grid_ids = grid_ids.cpu().numpy()
            for i, index in enumerate(grid_ids):
                emb[index] = pred[i]
            
    # print('loss:%.3f'%np.mean(losses))
    return np.mean(losses), emb


def sde_train(config, workdir):
    """Runs the training pipeline of molecule generation.

    Args:
        config: Configuration to use.
        workdir: Working directory for checkpoints and TF summaries.
            If this contains checkpoint training will be resumed from the latest checkpoint.
    """

    ### Ignore info output by RDKit
    RDLogger.DisableLog('rdApp.error')
    RDLogger.DisableLog('rdApp.warning')

    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    tb_dir = os.path.join(workdir, "tensorboard")
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)
    writer = tensorboard.SummaryWriter(tb_dir)

    # Initialize model.
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    score_model = mutils.create_model(config)
    score_model = score_model.to(config.device)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    # Create checkpoints directly
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(os.path.dirname(checkpoint_meta_dir)):
        os.makedirs(os.path.dirname(checkpoint_meta_dir))
    # Resume training when intermediate checkpoints are detected
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state['step'])


    #load dataset
   
    data_dir = "./data/"
    data_loader = Dataset(data_dir, max_nodes = config.data.max_node, batch_size = config.training.batch_size)
    train_data, test_data, train_ids, sample_ids  = data_loader.split_and_preprocess_data()
       
    # 应用转换
    
    transform = T.Compose([
    T.ToDevice(config.device)
    ])
    train_data = [transform(data) for data in train_data]
    test_data = [transform(data) for data in test_data] 


    # DataLoader

    train_dataloader = DataLoader(train_data, batch_size= config.training.batch_size, shuffle=True)
    pre_train_dataloader = DataLoader(train_data, batch_size= config.batch_size_pre, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size= config.training.eval_batch_size, shuffle=False)
    pre_test_dataloader = DataLoader(test_data, batch_size=config.batch_size_pre, shuffle = False)
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        #Variational Pathwise Stochastic Differential Equation
        atom_sde = sde_lib.VPSDE(beta_min=config.model.node_beta_min, beta_max=config.model.node_beta_max,
                                 N=config.model.num_scales)
        bond_sde = sde_lib.VPSDE(beta_min=config.model.edge_beta_min, beta_max=config.model.edge_beta_max,
                                 N=config.model.num_scales)
        sampling_eps = 1e-3
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_loss((atom_sde, bond_sde), train=True, optimize_fn=optimize_fn,
                                       reduce_mean=reduce_mean, continuous=continuous,
                                       likelihood_weighting=likelihood_weighting)
    eval_step_fn = losses.get_step_loss((atom_sde, bond_sde), train=False, optimize_fn=optimize_fn,
                                      reduce_mean=reduce_mean, continuous=continuous,
                                      likelihood_weighting=likelihood_weighting)

    sampling_atom_shape = (config.training.eval_batch_size, config.data.max_node, config.data.atom_channels)
    sampling_bond_shape = (config.training.eval_batch_size, config.data.bond_channels,
                            config.data.max_node, config.data.max_node)
    sampling_fn = sampling.get_mol_sampling_fn(config, atom_sde, bond_sde, sampling_atom_shape, sampling_bond_shape,
                                                inverse_scaler, sampling_eps)
    
    cond_pred_model = ConditionNetwork(config, dropout_rate= config.model.dropout_pre).to(config.device)
    opt_nn = torch.optim.Adam(cond_pred_model.parameters(), lr=config.optim.lr2,  weight_decay=config.optim.weight_decay)# 定义优化器，包括权重衰减（L2正则化）
    
    
    for it in tqdm(range(1, 1 + config.training.epoch)):
        loss_epoch = train_cond_model(cond_pred_model, pre_train_dataloader, opt_nn, scaler)
        logging.info('epoch:{} pre_loss: {:.6f}'.format(it,loss_epoch))
        
        if it % 5 == 0:
            eval_loss, _ = eval_cond_model(cond_pred_model, pre_test_dataloader, scaler)
            logging.info('epoch:{} test_loss: {:.6f}'.format(it,eval_loss))
            
        if it in [40, 60]:
            torch.save(cond_pred_model.state_dict(), checkpoint_dir + "model_{}.pth".format(it))
       
            


    num_train_steps = config.training.n_iters
  
    node_list = pickle.load(open(os.path.join(data_dir, 'nodes_dict.pkl'), 'rb')) 
    

    logging.info("Starting training loop at step %d." % (initial_step,))

    for step in range(initial_step, num_train_steps + 1):
        for graphs in train_dataloader:
            batch = dense_mol(graphs, scaler)
            model = state['model']
            optimizer = state['optimizer']
            optimizer.zero_grad()
            loss = train_step_fn(model, cond_pred_model, batch) 
            logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))
            writer.add_scalar("training_loss", loss, step)
            loss.backward()
            optimize_fn(optimizer, model.parameters(), step=state['step']) 
            
            state['step'] += 1
            state['ema'].update(model.parameters())

            if step % config.training.freq == 0:
                for graph in test_dataloader:
                    test_batch = dense_mol(graph, scaler)
                    node_feat, node_mask, bond_feat, bond_mask, house_node_mask, urban_attr,\
                        demand, pp, grid_ids = test_batch
                    with torch.no_grad():
                        ema.store(model.parameters())
                        ema.copy_to(model.parameters())
                        eval_loss = eval_step_fn(model, cond_pred_model, test_batch) 

                        ema.restore(model.parameters())
                        logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))

            if step % config.training.freq == 0:
                # Save the checkpoint.
                save_step = step // config.training.freq
                save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}_lr{config.optim.lr}_hiddensize{config.model.hidden}_layer{config.model.num_hybrid_layers}.pth'), state)

            


def evaluate(config, workdir, eval_folder="eval"):
    """Evaluate trained models.

    Args:
        config: Configuration to use.
        workdir: Working directory for checkpoints.
        eval_folder: The subfolder for storing evaluation results. Default to "eval".
    """


    # Create directory to eval_folder
    eval_dir = os.path.join(workdir, eval_folder)
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)



    data_dir = "/data/"
    data_loader = Dataset(data_dir, max_nodes = config.data.max_node, batch_size = config.training.batch_size)
    train_data, test_data, train_ids, sample_ids  = data_loader.split_and_preprocess_data()
    
    transform = T.Compose([
    T.ToDevice(config.device)
    ])
   
    test_data = [transform(data) for data in test_data] 

    # DataLoader
    node_list = pickle.load(open(os.path.join(data_dir, 'nodes_dict.pkl'), 'rb')) 
    test_dataloader = DataLoader(test_data, batch_size=config.eval.batch_size, shuffle=False, drop_last=True)
    pre_test_dataloader = DataLoader(test_data, batch_size=config.batch_size_pre, shuffle=False, drop_last=True)

    # Creat data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Initialize model
    score_model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    cond_pred_model = ConditionNetwork(config, dropout_rate= config.model.dropout).to(config.device)
    pre_train_dir = os.path.join(workdir, f"checkpointsmodel_{60}.pth")

    cond_pred_model.load_state_dict(torch.load(pre_train_dir))
    cond_pred_model.eval()
   
    loss, _ =eval_cond_model(cond_pred_model, pre_test_dataloader, scaler)
    
    print('loss:%.5f'%loss)

    checkpoint_dir = os.path.join(workdir, "checkpoints")
   
    atom_sde = sde_lib.VPSDE(beta_min=config.model.node_beta_min, beta_max=config.model.node_beta_max,
                                N=config.model.num_scales)
    bond_sde = sde_lib.VPSDE(beta_min=config.model.edge_beta_min, beta_max=config.model.edge_beta_max,
                                N=config.model.num_scales)
    sampling_eps = 1e-3

    sampling_atom_shape = (config.eval.batch_size, config.data.max_node, config.data.atom_channels)
    sampling_bond_shape = (config.eval.batch_size, config.data.bond_channels,
                            config.data.max_node, config.data.max_node)
    sampling_fn = sampling.get_mol_sampling_fn(config, atom_sde, bond_sde, sampling_atom_shape, sampling_bond_shape,
                                                inverse_scaler, sampling_eps)

    # Begin evaluation sample
    begin_ckpt = config.eval.begin_ckpt 
    logging.info("begin checkpoint: %d" % (begin_ckpt,))
    sample_dict ={}
    edge_dict = {}

    for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
        waiting_message_printed = False
        ckpt_filename = os.path.join(checkpoint_dir, f"checkpoint_{ckpt}_lr{config.optim.lr}_hiddensize{config.model.hidden}_layer{config.model.num_hybrid_layers}.pth")
    
        while not os.path.exists(ckpt_filename):
            if not waiting_message_printed:
                logging.warning(f"Waiting for the arrival of checkpoint {ckpt}")
                waiting_message_printed = True
            time.sleep(5)
        try:
            state = restore_checkpoint(ckpt_filename, state, device=config.device)
        except Exception as e:
            logging.error(f"Error occurred while restoring checkpoint: {e}")
        
        ema.copy_to(score_model.parameters())

        # Generate samples and store
        with torch.no_grad():
            for graph in test_dataloader:
                test_batch = dense_mol(graph, scaler)
                node_feat, node_mask, bond_feat, bond_mask, house_node_mask, urban_attr,\
                    demand, pp, grid_ids = test_batch
                atom_sample, bond_sample, sample_steps, sample_nodes, atom_mask, bond_mask,\
                            = sampling_fn(score_model, cond_pred_model, test_batch, node_list)
                grid_ids = grid_ids.cpu().numpy()
                sample_grids = atom_sample.detach().cpu().numpy()
                bond_sample = bond_sample.detach().cpu().numpy()
                for i, index in enumerate(grid_ids):
                    sample_dict[index] = sample_grids[i][:sample_nodes[i],:]
                    edge_dict[index] = bond_sample[i][:,:sample_nodes[i], :sample_nodes[i]]
            pickle.dump(sample_dict, open(os.path.join(eval_dir, "sample_node_{}.pkl".format(ckpt)), 'wb'))
            pickle.dump(edge_dict, open(os.path.join(eval_dir, "sample_edge_{}.pkl".format(ckpt)), 'wb')) 
             

        # evaluate
        logging.info('the nunber of sample grids: %d' % len(sample_dict))
   



