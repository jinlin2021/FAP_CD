import ml_collections
import torch


def get_config():
    config = ml_collections.ConfigDict()

    # training
    config.training = training = ml_collections.ConfigDict()
    training.sde = 'vpsde'
    training.continuous = True
    training.reduce_mean = False

    training.batch_size = 8
    
    training.eval_batch_size = 2
    training.n_iters = 2000000
    training.snapshot_freq = 1000  
    training.freq = 20 
    training.eval_freq = 5 
    ## produce samples at each snapshot.
    training.snapshot_sampling = True
    training.likelihood_weighting = False
    training.input_dim = 14
    # training.hidden_dim = 128
    training.output_dim = 14
    training.epoch = 2

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.method = 'dpm3'  
    sampling.predictor = 'euler_maruyama'
    sampling.corrector = 'none'
    sampling.rtol = 1e-5
    sampling.atol = 1e-5
    sampling.ode_method = 'rk4'
    sampling.ode_step = 0.01 #0.01  

    sampling.n_steps_each = 1
    sampling.noise_removal = True
    sampling.probability_flow = False
    sampling.atom_snr = 0.16
    sampling.bond_snr = 0.16
    sampling.vis_row = 4
    sampling.vis_col = 4

    sampling.batch_size = 32

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.begin_ckpt = 15
    evaluate.end_ckpt = 40
    evaluate.batch_size = 32  # 1024
    evaluate.enable_sampling = True
    evaluate.num_samples = 10000
    evaluate.mmd_distance = 'RBF'
    evaluate.max_subgraph = False
    evaluate.save_graph = False
    evaluate.nn_eval = False
    evaluate.nspdk = False

    # data
    config.data = data = ml_collections.ConfigDict()
    data.centered = True
    data.dequantization = False

    data.root = 'data'
    data.name = 'ZINC250K'
    data.split_ratio = 0.8
    # data.max_node = 38
    data.max_node = 400  # 400
    data.atom_channels = 14
    data.bond_channels = 1
    data.norm = (0.5, 1.0)

    

    # model
    config.model = model = ml_collections.ConfigDict()
    model.name = 'AF2CG'
    model.ema_rate = 0.9999
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.hidden = 128
    model.size =  64  
    model.num_hybrid_layers = 4  
    model.conditional = True
    model.embedding_type = 'positional'
    model.rw_depth = 20
    model.graph_layer = 'GCN'
    model.edge_th = -1.
    model.heads = 8
    model.dropout_pre = 0.1
    model.dropout = 0.3 #0.1
    

    model.num_scales = 200 #1000  # SDE total steps (N)
    model.sigma_min = 0.01
    model.sigma_max = 50
    model.node_beta_min = 0.1
    model.node_beta_max = 20.
    model.edge_beta_min = 0.1
    model.edge_beta_max = 20.

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 1e-3  
    optim.optimizer = 'Adam'
    optim.lr = 3e-4  
    optim.lr2 = 1e-5  
    optim.beta1 = 0.9 
    optim.eps = 1e-8 
    optim.warmup = 1000  
    optim.grad_clip = 1.  

    config.seed = 42
    config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    config.GTN_x_hiddim =  128 
    config.GTN_e_hiddim =  128 
    config.GTN_dim_ffX = 32
    config.GTN_dim_ffE =  32
    config.batch_size_pre = 32
    

    return config
