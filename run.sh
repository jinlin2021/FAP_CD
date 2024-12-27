
CUDA_VISIBLE_DEVICES= 0,1 nohup python main.py --config configs/config.py --mode train --workdir exp/output1 --config.training.batch_size 8 --config.batch_size_pre 6 --config.training.eval_batch_size 8 --config.sampling.method dpm3 --config.sampling.ode_step 200 --config.training.n_iters 1000 


