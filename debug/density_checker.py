import pickle
import os
import dask.bag as db
from collections import defaultdict
from scipy.sparse import coo_matrix
from scipy.sparse import save_npz

TIME_INTERVAL = 5

class Density_Distribution:
    def __init__(self, data):
        self.data = data
        self.density_distribution = self.process_data_parallel()

    def process_chunk(self, chunk):
        count_dict = defaultdict(int)
        for traj in chunk:
            for time, edge_id in traj:
                count_dict[(time // TIME_INTERVAL, edge_id)] += 1
        return [count_dict]

    def merge_dicts(self, dicts):
        merged_dict = defaultdict(int)
        for d in dicts:
            for key, value in d.items():
                merged_dict[key] += value
        return merged_dict

    def process_data_parallel(self):
        bag = db.from_sequence(self.data, npartitions=os.cpu_count())

        counts = bag.map_partitions(self.process_chunk).compute()

        merged_counts = self.merge_dicts(counts)

        rows, cols, values = zip(*((x, y, count) for (x, y), count in merged_counts.items()))
        max_x = max(rows) + 1
        max_y = max(cols) + 1
        sparse_matrix = coo_matrix((values, (rows, cols)), shape=(max_x, max_y))
        
        return sparse_matrix
    
    def get_distribution(self):
        return self.density_distribution

if __name__ == "__main__":
    data = []
    for data_type in ['train', 'val', 'test']:
        with open(f'data_provider/data/shenzhen_8_6/shenzhen_{data_type}_traj.pkl', 'rb') as file:
            data += pickle.load(file)
    density_distribution = Density_Distribution(data[:1])
    save_npz('dist.npz', density_distribution.get_distribution())
    