from matplotlib import pyplot as plt
import csv
import numpy as np
from lxml import etree
import ast
import pickle
from pyspark.sql import SparkSession
from tqdm import tqdm
from matplotlib.collections import LineCollection
from pyspark.sql.functions import col, sum, when
import os

class Checker:
    def __init__(self, ):
        self.car_passing_sparsity_check()
        pass

    def car_passing_sparsity_check(self, time_range=[39600, 39900]):
        # 12.31 检查车辆密度是否存在稀疏度 
        direction_dict = {
            0: 'Left',
            1: 'Straight',
            2: 'Right',
        }
        color_dict = {
            0: 'red',
            1: 'blue',
            2: 'green',
        }

        path = 'debug/shenzhen_8_6_passing_count.parquet'
        df = self.spark.read.parquet(path)
        df = df.filter(col('time').between(time_range[0], time_range[1])) 
            # .withColumn('passing_count', when(col('passing_count') > 0, 1).otherwise(col('passing_count')))
        edge_list = df.select('edge_id').distinct().rdd.map(lambda x: x[0]).collect()
        
        if not os.path.exists('fig/shenzhen_8_6/passing_count_visualize'):
            os.makedirs('fig/shenzhen_8_6/passing_count_visualize')

        for edge_id in tqdm(edge_list):
            plt.figure(figsize=(12, 6))
            for direction in direction_dict:
                # direction_df = df.filter((col('edge_id') == edge_id) & (col('direction') == direction)) \
                #     .drop('direction') \
                #     .sort('time')
                direction_df = df.filter((col('edge_id') == edge_id) & (col('direction') == direction)) \
                    .sort('time')
                pdf = direction_df.toPandas()
            
                plt.scatter(pdf['time'], 
                        pdf['direction'], 
                        color=color_dict[direction],
                        label=direction_dict[direction])

            plt.title(f'Passing Count over Time on Edge {edge_id}')
            plt.xlabel('Time')
            plt.ylabel('Density')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()

            save_path = f'fig/shenzhen_8_6/passing_count_visualize/edge {edge_id}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()