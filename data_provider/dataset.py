from torch.utils.data import Dataset
import pandas as pd
import os
import ast
import numpy as np
import subprocess

class Shenzhen(Dataset):
    def __init__(self, root_path, flag, window_size=2, time_range=['00:00:00', '23:59:59'], time_unit=5):
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

        self.__read_original_data(root_path)

    def __read_original_data(self, root_path):
        # unit time is not defined well and may cause error
        path_dict = {
            'train': os.path.join(root_path, f"shenzhen_train_traj.txt"),
            'val': os.path.join(root_path, f"shenzhen_val_traj.txt"),
            'test': os.path.join(root_path, f"shenzhen_test_traj.txt"),
        }

        # get the total number of data in flag dataset        
        result = subprocess.run(['wc', '-l', path_dict[self.flag]], stdout=subprocess.PIPE, text=True)
        num_data = int(result.stdout.split()[0])

        # get the total nubmer of data in train, val, test
        # num_data = 0
        # for flag in ['train', 'val', 'test']:
        #     result = subprocess.run(['wc', '-l', path_dict[flag]], stdout=subprocess.PIPE, text=True)
        #     num_data += int(result.stdout.split()[0])

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
                break
        for key in traffic_dict:
            traffic_dict[key] /= num_data / 1e4  # normalize the traffic

        # create x, y from file
        x_data = [] # (prev_loc, curr_loc, next_loc, prev_loc_traffic, curr_loc_traffic, next_loc_traffic, stay_time, curr_time)
        y_data = [] # 0 or 1
        with open(path_dict[self.flag], 'r') as f:
            aligned_trajs = []
            # aligned_trajs = np.full((len(lines), self.time_range[1] - self.time_range[0]), 0)
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
                        continue
                    else:
                        stay_time[i] = stay_time[i-1] + 1
                a_x_data = [route_context_dict[edge].copy() for i, edge in enumerate(duplicate_loc)]
                for i in range(len(a_x_data)):
                    route_context = a_x_data[i]
                    other_feature = [traffic_dict.get((time[i], edge), 0) for edge in route_context] + [stay_time[i]] + [time[i]]
                    a_x_data[i] += other_feature
                
                x_data += a_x_data
                a_y_data = (duplicate_loc != np.append(duplicate_loc[1:], 0)).tolist()
                y_data += a_y_data

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        return self.data[idx:idx+self.window_size]

if __name__ == "__main__":
    sz_data = Shenzhen(root_path='data_provider/data/shenzhen_8_6', flag='test', time_range=['13:00:00', '13:30:00'], time_unit=5, window_size=2)
