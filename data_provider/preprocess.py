from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import os
import re
import time
import numpy as np
import pandas as pd
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
            # self.traj_to_datapoints(data_type=data_type)   
            pass

        self.get_directions_of_edges()

    def get_directions_of_edges(self):
        edg_xml_file = "data_provider/data/shenzhen_8_6/edge_sumo.edg.xml"
        nod_xml_file = "data_provider/data/shenzhen_8_6/node_sumo.nod.xml"
        
        edges_df = pd.read_xml(edg_xml_file)
        nodes_df = pd.read_xml(nod_xml_file)

        edges_df = self.spark.createDataFrame(edges_df)
        nodes_df = self.spark.createDataFrame(nodes_df)

        nodes_dict = dict(nodes_df.rdd.map(lambda row: 
                (int(row.id), (float(row.x), float(row.y)))
            ).collect())
        nodes_broadcast = self.spark.sparkContext.broadcast(nodes_dict)

        def calculate_edge_info(row):
            """计算边的方向向量和坐标信息"""
            nodes = nodes_broadcast.value
            from_node = int(row['from'])
            to_node = int(row.to)
            
            # 获取坐标
            from_coord = nodes[from_node]  # (x1, y1)
            to_coord = nodes[to_node]      # (x2, y2)
            
            # 计算方向向量
            vector = np.array([
                to_coord[0] - from_coord[0],
                to_coord[1] - from_coord[1]
            ])
            norm = np.linalg.norm(vector)
            direction = vector / norm if norm > 1e-10 else np.array([0, 0])
            
            # 返回原有属性和新属性
            return {
                # 原有属性
                'id': int(row.id) - 1,  # very very important! There is a gap between the id of edg.xml and the id of traj file.
                'from': from_node,
                'to': to_node,
                'numLanes': int(row.numLanes),
                'speed': float(row.speed),
                'priority': int(row.priority),
                'length': float(row.length),
                # 新增属性
                'direction': direction.tolist(),  # 转换为list以便序列化
                'from_x': float(from_coord[0]),
                'from_y': float(from_coord[1]),
                'to_x': float(to_coord[0]),
                'to_y': float(to_coord[1])
            }

        # 应用转换
        edges_rdd = edges_df.rdd.map(calculate_edge_info)
        
        result_df = self.spark.createDataFrame(edges_rdd)
        result_df.show()
        result_df.write.mode('overwrite').parquet("data_provider/data/shenzhen_8_6/edges_with_direction.parquet")
        return result_df

    def get_counts_of_one_way_edges_and_two_way_edges(self):
        edg_xml_file = "data_provider/data/shenzhen_8_6/edge_sumo.edg.xml"
        df = pd.read_xml(edg_xml_file).set_index('id')
        print("len(df):")
        print(len(df))
        
        df['edge_pair'] = df.apply(lambda row: tuple(sorted([row['from'], row['to']])), axis=1)
        edge_counts = df.groupby('edge_pair').size().reset_index(name='edge_count')

        unique_counts = edge_counts['edge_count'].unique()
        print("unique_counts:")
        print(unique_counts)

        counts_of_edge_counts = edge_counts.groupby('edge_count').size().reset_index(name='count')
        print("counts_of_edge_counts:")
        print(counts_of_edge_counts)

    def traj_to_datapoints(self, slice_len=300, data_type='test'):
        file_path = f"data_provider/data/shenzhen_8_6/shenzhen_{data_type}_traj.pkl"
        save_path = f"data_provider/data/shenzhen_8_6/shenzhen_{data_type}_traj.parquet"
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        
        # (time, edge_id)
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

class Preprocess_Traffic_Light:
    def __init__(self):
        pass

    def get_traffic_light_data(self):
        file_path = "data_provider/data/traffic_light/processed_matrix/signal_matrix.npz"
        data = np.load(file_path)
        dict_data = dict(data)
        for key, value in dict_data.items():
            print(key, value.shape)
        pass

    def get_lane_type_vector(self):
        file_path = "data_provider/data/traffic_light/processed_matrix/lane_type_vector.pkl"
        with open(file_path, 'rb') as file:
            lane_type_vector = pickle.load(file)
        return lane_type_vector


if __name__ == "__main__":
    preprocess = Preprocess_Shenzhen(data_types=['test'])