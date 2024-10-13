import torch
from config import get_cfg, get_model
from ma_model import SpatialTemporalCrossMultiAgentModel

if __name__ == "__main__":
    cfg = get_cfg()
    print(cfg)
    model = get_model(cfg)
    print(model)