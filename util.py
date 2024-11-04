from timepredmodel import TimePredModel, MLPModel

def get_model(cfg):
    if cfg.name == 'mlp':
        model = MLPModel(
            input_dim=cfg.input_dim, 
            hidden_dims=cfg.hidden_dims, 
            output_dim=cfg.output_dim
        )
    else:
        model = TimePredModel(
            vocab_size=cfg.vocab_size,
            route_dim=cfg.route_dim, 
            space_dim=cfg.space_dim, 
            time_dim=cfg.time_dim, 
            route_hidden=cfg.route_hidden, 
            state_hidden=cfg.state_hidden, 
            window_size=cfg.window_size, 
            block_dims=cfg.block_dims
        ) 
    return model