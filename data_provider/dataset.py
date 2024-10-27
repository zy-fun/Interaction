from torch.utils.data import Dataset
import pandas as pd
import os
import ast
import numpy as np
import subprocess

class Shenzhen(Dataset):
    def __init__(self, root_path, flag, window_size=3, time_range=['00:00:00', '23:59:59'], time_unit=5):
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
    
    def print(self):
        print(self.date)

    def __read_original_data(self, root_path):
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

        with open(path_dict[self.flag], 'r') as f:
            aligned_trajs = []
            # aligned_trajs = np.full((len(lines), self.time_range[1] - self.time_range[0]), 0)
            for line in f:
                time, loc = zip(*[ast.literal_eval(traj_point) for traj_point in line.split()[1:]])
                # traj_start_time = time[0] // self.time_unit - self.time_range[0]
                # traj_end_time = time[-1] // self.time_unit - self.time_range[0]
                # aligned_traj = np.full(self.time_range[1] - self.time_range[0], 0) 
                # aligned_traj[traj_start_time: traj_end_time + 1] = np.array(loc)
                # print(aligned_traj)
                # print(list(aligned_traj))
                # print(traj_start_time, traj_end_time)
                # break
                
                duplicate_loc = np.array(loc)
                change_indices = np.append([0], np.where(np.diff(loc) != 0)[0] + 1)
                unique_loc = duplicate_loc[change_indices]
                loc_counts = np.append(np.diff(change_indices), len(loc) - change_indices[-1])
                
                print(loc)
                print(unique_loc)
                print(loc_counts)
                exit()
            


    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        return self.data[idx:idx+self.window_size]

if __name__ == "__main__":
    sz_data = Shenzhen(root_path='data_provider/data/shenzhen_8_6', flag='test', time_range=['13:00:00', '13:30:00'], time_unit=5)
