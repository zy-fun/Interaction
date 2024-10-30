from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import ast
import numpy as np
import subprocess
from lxml import etree

class Dataset_Shenzhen(Dataset):
    def __init__(self, root_path, flag, window_size=2, time_range=['00:00:00', '23:59:59'], time_unit=5, ):
        assert flag in ['train', 'val', 'test']
        self.window_size = window_size
        self.flag = flag
        self.root_path = root_path

        # compute time_range
        start_time = [int(t) for t in time_range[0].split(':')]
        start_time = start_time[0] * 3600 + start_time[1] * 60 + start_time[2]
        end_time = [int(t) for t in time_range[1].split(':')]
        end_time = end_time[0] * 3600 + end_time[1] * 60 + end_time[2]
        self.time_range = [start_time // time_unit, end_time // time_unit]
        self.time_unit = time_unit

        self.x_data = []    # [route_id(window_size), route_feature(window_size), space_state(window_size), time_stay, curr_time]
        self.y_data = []    # T or F
        self.__read_original_data(root_path)

    def __read_original_data(self, root_path):
        # unit time is not defined well and may cause error
        path_dict = {
            'train': os.path.join(root_path, f"shenzhen_train_traj.txt"),
            'val': os.path.join(root_path, f"shenzhen_val_traj.txt"),
            'test': os.path.join(root_path, f"shenzhen_test_traj.txt"),
            'edge': os.path.join(root_path, f"edge_sumo.edg.xml"),
        }

        # get the total number of data in flag dataset        
        result = subprocess.run(['wc', '-l', path_dict[self.flag]], stdout=subprocess.PIPE, text=True)
        num_data = int(result.stdout.split()[0])

        # calculate the total traffic at each time and location
        traffic_dict = {}   # (time, loc) -> traffic
        with open(path_dict[self.flag], 'r') as f:
            for line in f:
                trajs = [ast.literal_eval(traj_point) for traj_point in line.split()[1:]]
                for t, loc in trajs:
                    t = t // self.time_unit
                    if (t, loc) in traffic_dict:
                        traffic_dict[(t, loc)] += 1
                    else:
                        traffic_dict[(t, loc)] = 1

        # normalize the traffic dict
        for key in traffic_dict:
            traffic_dict[key] /= num_data / 1e4  

        # create edge_features from file
        edge_features = {}
        edges_tree = etree.parse(path_dict['edge'])
        edges = edges_tree.getroot()
        for edge in edges.xpath('./edge'):
            edge_id = int(edge.get('id')) - 1
            edge_features[edge_id] = float(edge.get('length'))

        # create x, y from file
        with open(path_dict[self.flag], 'r') as f:
            for line in f:
                time, duplicate_loc = zip(*[ast.literal_eval(traj_point) for traj_point in line.split()[1:]])
                time = np.array(time) // self.time_unit
                traj_start_time = time[0]
                duplicate_loc = np.array(duplicate_loc)
                change_indices = np.append([0], np.where(np.diff(duplicate_loc) != 0)[0] + 1)
                route = list(duplicate_loc[change_indices]) + [0] * (self.window_size - 1)
                route_context_dict = {route[i]:route[i:i+self.window_size] for i in range(len(route) - self.window_size + 1)}
                stay_time = [0] * len(duplicate_loc)
                for i in range(len(duplicate_loc)):
                    if i == 0 or duplicate_loc[i] != duplicate_loc[i-1]:
                        pass
                    else:
                        stay_time[i] = stay_time[i-1] + 1
                a_x_data = [route_context_dict[edge].copy() for edge in duplicate_loc]
                for i in range(len(a_x_data)):
                    route_context = a_x_data[i]
                    route_feature = [float(edge_features.get(edge, 0)) for edge in route_context]
                    space_state = [traffic_dict.get((time[i], edge), 0) for edge in route_context]
                    time_stay = [stay_time[i]]
                    curr_time = [time[i]]
                    a_x_data[i] += route_feature + space_state + time_stay + curr_time
                
                self.x_data += a_x_data
                a_y_data = (duplicate_loc != np.append(duplicate_loc[1:], 0)).tolist()
                self.y_data += a_y_data

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

if __name__ == "__main__":
    sz_data = Dataset_Shenzhen(root_path='data_provider/data/shenzhen_8_6', flag='test', time_range=['13:00:00', '13:30:00'], time_unit=5, window_size=2)
    data_loader = DataLoader(sz_data, batch_size=32, shuffle=True)
    for x, y in data_loader:
        print(x)
        print(y)
        break 