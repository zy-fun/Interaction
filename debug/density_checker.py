import pickle
import os
import dask.bag as db
from collections import defaultdict
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, count
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import time
import numpy as np
from functools import reduce
import pyspark.sql

TIME_INTERVAL = 5
    
class Density_Checker:
    def __init__(self):
        self.spark = SparkSession.builder \
            .config("spark.executor.memory", "8g") \
            .config("spark.driver.memory", "8g") \
            .config("spark.memory.offHeap.enabled", "true") \
            .config("spark.memory.offHeap.size", "4g") \
            .master("local") \
            .appName("get_num_of_data_points"). \
            getOrCreate()

        self.get_distribution_by_lane(data_types=['train', 'val', 'test'])
        pass

    def get_distribution_by_lane(self, data_types=['train', 'val', 'test']):
        dfs = []
        for data_type in data_types:
            df = self.spark.read.parquet(f'data_provider/data/shenzhen_8_6/shenzhen_{data_type}_traj.parquet')
            dfs.append(df)
        df = reduce(pyspark.sql.DataFrame.union, dfs)

        # synchronize the time
        df = df.withColumn('time', (col('time') / TIME_INTERVAL).cast('int') * TIME_INTERVAL)
        df = df \
            .select('time', 'edge_id', 'direction') \
            .groupby('time', 'edge_id', 'direction') \
            .agg(count('*').alias('density'))

        df.write.mode('overwrite').parquet('debug/shenzhen_8_6_density_by_lane.parquet')
        df.show()

    def check_direction(self):
        pass

    def visualize_traffic_density(self, dist, time_range=(5760, 6480)):
        # 1. calculate the density of each edge
        # ((time, edge_id), count)
        dist_rdd = self.spark.sparkContext.parallelize(dist, numSlices=os.cpu_count())
        dist_rdd = dist_rdd.filter(lambda x: x[0][0] >= time_range[0] and x[0][0] <= time_range[1]) \
            .map(lambda x: (x[0][1], x[1])) \
            .reduceByKey(lambda a, b: a + b)

        values = dist_rdd.values()
        max_value = values.max()

        def normalize(value):
            normalized = value / max_value
            return 0.3 + normalized * 0.7

        dist_rdd = dist_rdd.map(lambda x: (x[0], normalize(x[1])))
        density = np.zeros(dist_rdd.keys().max() + 1)
        for k, v in dist_rdd.collect():
            density[k] = v        

        # 2. get the coordinates of each edge
        edge_loc_file = "data_provider/data/shenzhen_8_6/edge_loc.csv"
        edge_loc_df = self.spark.read.option("header", "true") \
            .option("inferSchema", "true") \
            .csv(edge_loc_file) \
            .orderBy("index")
        edge_loc_df = edge_loc_df.withColumn("index", (col("index") - 1))
        
        segments = edge_loc_df.rdd.map(
            lambda row: [(row.lon1, row.lat1), (row.lon2, row.lat2)]
        ).collect()

        # 3. plot the density
        fig, ax = plt.subplots(figsize=(32, 20))
        
        cmap = plt.cm.Reds  # 使用原来的 Reds 色图
        my_cmap = cmap.copy()
        my_cmap.set_under('grey')  # 设置小于 vmin 的颜色为灰色

        lc = LineCollection(segments,
                        cmap=my_cmap,
                        norm=plt.Normalize(vmin=1e-6, vmax=1),  # 设置一个很小的正数作为最小值
                        linewidths=1 + np.array(density) * 3,
                        )

        # 将密度为0的值设置为比 vmin 更小的值
        density_masked = np.where(density > 0, density, -1)
        lc.set_array(density_masked)
        
        ax.add_collection(lc)
        
        ax.autoscale() 
        plt.title('Traffic Flow Visualization')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        plt.colorbar(lc, label='Flow')
        
        plt.grid(True, alpha=0.3)
        plt.axis('auto')
        plt.savefig('debug/density.png')

    def get_statistics(self, slice_len=1000, data_types=['val', 'test']):
        data = []
        for data_type in data_types:
            with open(f'data_provider/data/shenzhen_8_6/shenzhen_{data_type}_traj.pkl', 'rb') as file: 
                data += pickle.load(file) # word count
        
        statistics = {}
        
        # get number of data points
        rdd = self.spark.sparkContext.parallelize(data, numSlices=len(data) // slice_len)
        flattened_rdd = rdd.flatMap(lambda x: x)
        total_data_points_num = flattened_rdd.count()
        statistics['total_data_points_num'] = total_data_points_num

        # get avg length of trajectory
        rdd = self.spark.sparkContext.parallelize(data, numSlices=len(data) // slice_len)
        num_trajs = rdd.count()
        statistics['num_trajs'] = num_trajs

        avg_data_points_per_traj = total_data_points_num // num_trajs
        statistics['avg_data_points_per_traj'] = avg_data_points_per_traj

        return statistics


    def get_distribution_from_file(self, slice_len=1000, data_types=['val', 'test']):
        data = []
        for data_type in data_types:
            with open(f'data_provider/data/shenzhen_8_6/shenzhen_{data_type}_traj.pkl', 'rb') as file:
                data += pickle.load(file) 
        
        # word count
        spark = SparkSession.builder \
            .config("spark.executor.memory", "8g") \
            .config("spark.driver.memory", "8g") \
            .config("spark.memory.offHeap.enabled", "true") \
            .config("spark.memory.offHeap.size", "4g") \
            .master("local") \
            .appName("Tuple Word Count"). \
            getOrCreate()
        rdd = spark.sparkContext.parallelize(data, numSlices=len(data) // slice_len)

        flattened_rdd = rdd.flatMap(lambda x: x)

        result_rdd = flattened_rdd.map(lambda x: ((x[0] // TIME_INTERVAL, x[1]), 1)).reduceByKey(lambda a, b: a + b)
        dist = result_rdd.collect()
        spark.stop()
        return dist
    
    def get_sum_of_density(self, data):
        spark = SparkSession.builder \
            .config("spark.executor.memory", "8g") \
            .config("spark.driver.memory", "8g") \
            .config("spark.memory.offHeap.enabled", "true") \
            .config("spark.memory.offHeap.size", "4g") \
            .master("local") \
            .appName("Tuple Word Count"). \
            getOrCreate()
        rdd = spark.sparkContext.parallelize(data, numSlices=os.cpu_count())

        sum_of_data = rdd.map(lambda x: x[1]).reduce(lambda a, b: a + b)
        spark.stop()
        return sum_of_data
    
    def get_most_frequent(self, dist, n=100, reverse=False):
        spark = SparkSession.builder \
            .config("spark.executor.memory", "8g") \
            .config("spark.driver.memory", "8g") \
            .config("spark.memory.offHeap.enabled", "true") \
            .config("spark.memory.offHeap.size", "4g") \
            .master("local") \
            .appName("get_most_frequent"). \
            getOrCreate()

        rdd = spark.sparkContext.parallelize(dist, numSlices=os.cpu_count())
        sorted_rdd = rdd.sortBy(lambda x: x[1], ascending=reverse)

        top_n_rdd = sorted_rdd.zipWithIndex().filter(lambda x: x[1] < n).map(lambda x: x[0])
        result = top_n_rdd.collect()
        spark.stop()
        return result

    def resample_on_time(self, data, new_time_interval=60):
        spark = SparkSession.builder \
            .config("spark.executor.memory", "8g") \
            .config("spark.driver.memory", "8g") \
            .config("spark.memory.offHeap.enabled", "true") \
            .config("spark.memory.offHeap.size", "4g") \
            .master("local") \
            .appName("get_most_frequent"). \
            getOrCreate()
        
        rdd = spark.sparkContext.parallelize(dist, numSlices=os.cpu_count())
        rdd = rdd.filter(lambda x: x[0])

    def get_num_of_most_frequent(self, dist, threshold=1, reverse=False):
        spark = SparkSession.builder \
            .config("spark.executor.memory", "8g") \
            .config("spark.driver.memory", "8g") \
            .config("spark.memory.offHeap.enabled", "true") \
            .config("spark.memory.offHeap.size", "4g") \
            .master("local") \
            .appName("get_num_of_most_frequent"). \
            getOrCreate()

        rdd = spark.sparkContext.parallelize(dist, numSlices=os.cpu_count() * 3)
        rdd = rdd.filter(lambda x: x[1] >= threshold)
        num_of_most_frequent = rdd.count()
        result = rdd.collect()
        spark.stop()
        return result, num_of_most_frequent

    def get_num_of_least_frequent(self, dist, threshold=1, reverse=False):
        spark = SparkSession.builder \
            .config("spark.executor.memory", "8g") \
            .config("spark.driver.memory", "8g") \
            .config("spark.memory.offHeap.enabled", "true") \
            .config("spark.memory.offHeap.size", "4g") \
            .master("local") \
            .appName("get_num_of_least_frequent"). \
            getOrCreate()

        rdd = spark.sparkContext.parallelize(dist, numSlices=os.cpu_count() * 3)
        rdd = rdd.filter(lambda x: x[1] <= threshold)
        result = rdd.collect()
        num_of_least_frequent = rdd.count()
        spark.stop()
        return result, num_of_least_frequent

if __name__ == "__main__":
    starttime = time.time()
    data_types = ['train', 'val', 'test']

    checker = Density_Checker()
    # checker.view()

    # calculate number of data points
    # if not os.path.exists('debug/statistics.txt'):
    #     stat = checker.get_statistics(data_types=data_types)
    #     stat = str(stat)
    #     with open('debug/statistics.txt', 'w') as file:
    #         file.write(stat)
    # else:
    #     stat = open('debug/statistics.txt', 'r').read()

    # get the distribution
    # if os.path.exists('debug/dist.pkl'):
    #     with open('debug/dist.pkl', 'rb') as file:
    #         dist = pickle.load(file)
    # else:
    #     dist = checker.get_distribution_from_file(data_types=data_types)
    #     with open('debug/dist.pkl', 'wb') as file:
    #         pickle.dump(dist, file)
    
    # checker.visualize_traffic_density(dist)

    # result = checker.get_sum_of_density(dist)
    # print(result)
    
    # result = checker.get_most_frequent(dist, n=100)
    # with open('debug/dist_100.pkl', 'wb') as file:
    #     pickle.dump(result, file)
    # print(result)

    # least_frequent, num_of_least_frequent = checker.get_num_of_least_frequent(dist, threshold=10)
    # print(least_frequent)
    # print(num_of_least_frequent)

