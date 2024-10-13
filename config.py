from typing import List, Optional
from ma_model import SpatialTemporalCrossMultiAgentModel
import argparse
import torch

def get_model(cfg, load_from:Optional[str] = None):
    N = cfg.N
    device = cfg.device
    block_size, n_embd, n_head, n_layer, dropout = cfg.block_size, cfg.n_embd, cfg.n_head, cfg.n_layer, cfg.dropout
    n_hidden = cfg.n_hidden
    n_embed_adj = cfg.n_embed_adj
    vocab_size = cfg.vocab_size
    use_ne = cfg.use_ne
    use_ge = cfg.use_ge 
    use_adaLN = cfg.use_adaLN
    use_adjembed = cfg.use_adjembed
    postprocess = cfg.postprocess
    window_size = cfg.window_size

    use_model = cfg.use_model
    graph_embedding_mode = cfg.graph_embedding_mode
    if use_model=="sd":
        model = SpatialTemporalCrossMultiAgentModel(vocab_size, 
                                                    n_embd, 
                                                    n_hidden, 
                                                    n_layer, 
                                                    n_head, 
                                                    block_size,
                                                    n_embed_adj,
                                                    window_size=window_size,
                                                    dropout=dropout, 
                                                    use_ne=use_ne, 
                                                    use_ge=use_ge, 
                                                    device=device,
                                                    postprocess=postprocess,
                                                    use_adjembed=use_adjembed,
                                                    graph_embedding_mode=graph_embedding_mode
                                                    )
    else:
        raise NotImplementedError
    # elif use_model=="naive":
    #     model = NaiveMultiAgentLanguageModel(N,vocab_size, n_embd, n_layer,n_head,block_size, dropout,device=device)
    # else:    
    #     model = MultiAgentBigramLanguageModelWithoutAttention(N,vocab_size)

    model = model.to(device)
    
    if load_from is not None:
        model.load_state_dict(torch.load(load_from, map_location=device),strict=True)
    
    return model

def get_cfg(args: Optional[List[str]]=None)->dict:
    str2bool = lambda x: x.lower() in ['true', '1', 't','y','yes']
    
    parser = argparse.ArgumentParser()
    # Defining Scenario
    parser.add_argument('--expname', type=str, default='n', help='Name of the experiment')
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--setting', type=str, default='boston', choices=['boston', 'grid', 'paris', 'porto', 'beijing', 'jinan'])
    parser.add_argument('--new_dataloader', type=str2bool, default=True)
    parser.add_argument('--N', type=int, default=1)
    parser.add_argument('--grid_size', type=int, default=10)
    parser.add_argument('--enable_interaction', type=str2bool, default=True)
    parser.add_argument('--random_od', type=str2bool, default=True)
    
    parser.add_argument('--use_wandb', type=str2bool, default=True)
    
    # For data preprocess
    parser.add_argument('--vocab_size', type=int, default=101)
    parser.add_argument('--num_processes','-np', type=int, default=100) # not used by far
    parser.add_argument('--total_trajectories', type=int, default=50000)
    parser.add_argument('--with_closed', type=str2bool, default=False, help='Grid with closed points')
    parser.add_argument('--hop', type=int, default=2)
    parser.add_argument('--root_path', type=str, default='.', help='root path')
    parser.add_argument('--graph_path', type=str, default='graph/boston_100.pkl', help='graph path')
    parser.add_argument('--data_path', type=str, default='data/data_100.npy', help='data path')
    parser.add_argument('--length_path', type=str, default='data/valid_length_100.npy', help='valid length path')
    parser.add_argument('--distance_path', type=str, default='data/distance.npy', help='distance path')
    parser.add_argument('--od_per_graph', type=int, default=1000)
    parser.add_argument('--num_file', type=int, default=100)

    # For model
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--use_agent_mask', type=str2bool, default=True)
    parser.add_argument('--use_adj_mask', type=str2bool, default=False)
    parser.add_argument('--use_model', type=str, default='sd', choices=['sd', 'naive', 'noattention'])
    parser.add_argument('--use_ne', type=str2bool, default=True, help='Use Normalized Toekn Embedding')
    parser.add_argument('--use_ge', type=str2bool, default=False, help='Use Geolocation Embedding')
    parser.add_argument('--use_adaLN', type=str2bool, default=True, help='Use Adaptation Layernorm')
    parser.add_argument('--use_adjembed', type=str2bool, default=True, help='Adj embed from sratch')
    parser.add_argument('--postprocess', type=str2bool, default=False, help='Mul adj before softmax')
    parser.add_argument('--window_size', type=int, default=4, help='prefix length')

    parser.add_argument('--norm_position', type=str, default='prenorm', choices=['prenorm', 'postnorm'])
    parser.add_argument('--block_size', type=int, default=4)
    parser.add_argument('--n_embd', type=int, default=64)
    parser.add_argument('--n_embed_adj', type=int, default=16)
    parser.add_argument('--n_hidden', type=int, default=64)
    #parser.add_argument('--d_cross', type=int, default=64)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.0)
    
    # for training
    parser.add_argument('--save_model', type=str2bool, default=True)
    parser.add_argument('--use_dp', type=str2bool, default=False)
    parser.add_argument('--ddp_device_ids', type=str, default=None,help="Usage: --dp_device_ids='0,1,2,3'")
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_iters', type=int, default=3000)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--learning_rate_s2r', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--graph_embedding_mode', type=str, default='adaLN', choices=['adaLN', 'none', 'add', 'cross'])
    parser.add_argument('--iter_per_epoch', type=int, default=100)
    parser.add_argument('--finetune_load_from', type=str, default=None)

    # grad clip
    parser.add_argument('--use_ucbgc', type=str2bool, default=True)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--ucbgc_alpha', type=float, default=0.99)
    parser.add_argument('--ucbgc_beta', type=float, default=1.0)
    
    parser.add_argument('--eval_iters', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--save_interval', type=int, default=5000)
    
    # for eval
    parser.add_argument('--eval_load_from', type=str, default=None)
    
    cfg = parser.parse_args(args)
    if cfg.use_dp:
        assert cfg.dp_device_ids is not None, "Please specify the device ids for Data Parallel"
        cfg.dp_device_ids = list(map(int,cfg.dp_device_ids.split(',')))
        cfg.batch_size = cfg.batch_size * len(cfg.dp_device_ids)
        cfg.device = 'cuda'
    
    #     cfg = vars(parser.parse_args(args))
    # if cfg['use_dp']:
    #     assert cfg['dp_device_ids'] is not None, "Please specify the device ids for Data Parallel"
    #     cfg['dp_device_ids'] = list(map(int,cfg['dp_device_ids'].split(',')))
    #     cfg['batch_size'] = cfg['batch_size'] * len(cfg['dp_device_ids'])
    #     cfg['device'] = 'cuda'
    # # cfg['vocab_size'] = 101
    # # cfg['max_value'] = cfg['grid_size']
    
    return cfg