from pyspark.sql import SparkSession
import os
import re
import time
import pickle

class Preprocess_Shenzhen:
    def __init__(self, data_types=['train','val', 'test']):
        self.spark = SparkSession.builder \
            .config("spark.executor.memory", "8g") \
            .config("spark.driver.memory", "8g") \
            .config("spark.memory.offHeap.enabled", "true") \
            .config("spark.memory.offHeap.size", "4g") \
            .master("local") \
            .appName("Preprocess Shenzhen") \
            .getOrCreate()

        for data_type in data_types:
            # self.txt_to_pkl(data_type=data_type)
            self.traj_to_datapoints(data_type=data_type)   

    def traj_to_datapoints(self, slice_len=300, data_type='test'):
        file_path = f"data_provider/data/shenzhen_8_6/shenzhen_{data_type}_traj.pkl"
        save_path = f"data_provider/data/shenzhen_8_6/shenzhen_{data_type}_traj.parquet"
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        
        rdd = self.spark.sparkContext.parallelize(data, numSlices=len(data) // slice_len)
        
        def append_stay_time_field(traj):
            append_traj = []
            for i, point in enumerate(traj):
                if i > 0 and traj[i-1][1] == point[1]:
                    append_traj.append((*point, append_traj[-1][-1] + 1))
                else:
                    append_traj.append((*point, 0))
            return append_traj
        
        def append_about_to_move_field(traj):
            append_traj = []
            for i, point in enumerate(traj):
                if i < len(traj)-1 and traj[i+1][1] == point[1]:
                    append_traj.append((*point, 0))
                else:
                    append_traj.append((*point, 1))
            return append_traj

        def append_traj_id_field(x):
            traj, traj_id = x
            return [(*point, traj_id) for point in traj]

        def append_end_of_traj_field(traj):
            return [(*point, 1 if i == len(traj)-1 else 0) for i, point in enumerate(traj)]

        rdd = rdd.map(append_stay_time_field).map(append_about_to_move_field)
        rdd = rdd.zipWithIndex().map(append_traj_id_field)
        flattened_rdd = rdd.flatMap(append_end_of_traj_field)
            
        columns = ['time', 'edge_id', 'stay_time', 'about_to_move', 'traj_id', 'end_of_traj']
        df = flattened_rdd.toDF(columns)
        df.show()
        df.write.mode('overwrite').parquet(save_path)
        return data, df

    def txt_to_pkl(self, slice_len=1000, data_type='test'):
        file_path = f"data_provider/data/shenzhen_8_6/shenzhen_{data_type}_traj.txt"
        save_path = f"data_provider/data/shenzhen_8_6/shenzhen_{data_type}_traj.pkl"

        rdd = self.spark.sparkContext.textFile(file_path)
        rdd = rdd.map(lambda line: [int(i) for i in re.findall(r'\d+', line)[1:]])
        results = rdd.map(lambda traj: [(traj[i], traj[i+1]) for i in range(0, len(traj), 2)]).collect()

        with open(save_path, 'wb') as file:
            pickle.dump(results, file) 

if __name__ == "__main__":
    preprocess = Preprocess_Shenzhen(data_types=['test'])