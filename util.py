from timepred import TimePredModel

def get_model(cfg):
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