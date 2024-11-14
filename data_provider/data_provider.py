from torch.utils.data import DataLoader
from data_provider.dataset import Dataset_Shenzhen, miniDataset, DownSampledDataset, BinaryClassificationDataset

def get_dataloader(cfg, data_type: str):
    data_name = cfg.data_name
    root_path_dict = {
        'shenzhen_8_6': 'data_provider/data/shenzhen_8_6',
        'mini_data': 'data_provider/data/shenzhen_8_6'
    }
    root_path = root_path_dict.get(data_name, None)
 
    if data_name == 'mini_data':
        data = miniDataset(root_path, flag=data_type, window_size=cfg.window_size)
    elif data_name == 'shenzhen_8_6':
        data = Dataset_Shenzhen(root_path=root_path, flag=data_type, window_size=cfg.window_size)
    elif data_name == 'binary_classification':
        data = BinaryClassificationDataset(n_features=cfg.input_dim, data_type=data_type)

    if cfg.downsample:
        data = DownSampledDataset(data.x_data, data.y_data)

    data_loader = DataLoader(data, batch_size=cfg.batch_size, shuffle=cfg.shuffle)
    return data_loader