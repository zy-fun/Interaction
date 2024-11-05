from torch.utils.data import DataLoader
from data_provider.dataset import Dataset_Shenzhen, miniDataset, DownSampledDataset

def get_dataloader(cfg, data_type: str):
    data_name = cfg.data_name
    root_path_dict = {
        'shenzhen_8_6': 'data_provider/data/shenzhen_8_6',
        'mini_data': 'data_provider/data/shenzhen_8_6'
    }
    root_path = root_path_dict[data_name]
 
    if data_name == 'mini_data':
        data = miniDataset(root_path, flag=data_type, window_size=cfg.window_size)
    elif data_name == 'shenzhen_8_6':
        data = Dataset_Shenzhen(root_path=root_path, flag=data_type, window_size=cfg.window_size)

    if cfg.downsample:
        data = DownSampledDataset(data.x_data, data.y_data)

    data_loader = DataLoader(data, batch_size=cfg.batch_size, shuffle=True)
    return data_loader