from torch.utils.data import DataLoader
from data_provider.dataset import Dataset_Shenzhen



def get_dataloader(cfg, data_type: str):
    data_name = cfg.data_name
    root_path_dict = {
        'shenzhen_8_6': 'data_provider/data/shenzhen_8_6'
    }
    root_path = root_path_dict[data_name]

    dataset_dict = {
        'shenzhen_8_6': Dataset_Shenzhen
    }
    
    data = dataset_dict[data_name](root_path=root_path, flag=data_type, window_size=cfg.window_size)
    data_loader = DataLoader(data, batch_size=cfg.batch_size, shuffle=True)
    return data_loader