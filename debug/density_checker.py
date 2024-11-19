import pickle
import os
import dask.bag as db
from collections import defaultdict
from scipy.sparse import coo_matrix
from scipy.sparse import save_npz
from pyspark.sql import SparkSession
import time

TIME_INTERVAL = 5
    
class Density_Checker:
    def __init__(self):
        pass

    def get_statistics(self, slice_len=1000, data_types=['val', 'test']):
        data = []
        for data_type in data_types:
            with open(f'data_provider/data/shenzhen_8_6/shenzhen_{data_type}_traj.pkl', 'rb') as file: data += pickle.load(file) # word count
        spark = SparkSession.builder \
            .config("spark.executor.memory", "8g") \
            .config("spark.driver.memory", "8g") \
            .config("spark.memory.offHeap.enabled", "true") \
            .config("spark.memory.offHeap.size", "4g") \
            .master("local") \
            .appName("get_num_of_data_points"). \
            getOrCreate()
        
        statistics = {}
        
        # get number of data points
        rdd = spark.sparkContext.parallelize(data, numSlices=len(data) // slice_len)
        flattened_rdd = rdd.flatMap(lambda x: x)
        total_data_points_num = flattened_rdd.count()
        statistics['total_data_points_num'] = total_data_points_num

        # get avg length of trajectory
        rdd = spark.sparkContext.parallelize(data, numSlices=len(data) // slice_len)
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

if __name__ == "__main__":
    starttime = time.time()
    data_types = ['train', 'val', 'test']

    checker = Density_Checker()

    # calculate number of data points
    if not os.path.exists('debug/statistics.txt'):
        stat = checker.get_statistics(data_types=data_types)
        stat = str(stat)
        with open('debug/statistics.txt', 'w') as file:
            file.write(stat)
    else:
        stat = open('debug/statistics.txt', 'r').read()
    print(stat)

    # get the distribution
    if os.path.exists('debug/dist.pkl'):
        with open('debug/dist.pkl', 'rb') as file:
            dist = pickle.load(file)
    else:
        dist = checker.get_distribution_from_file(data_types=data_types)
        with open('debug/dist.pkl', 'wb') as file:
            pickle.dump(dist, file)
    
    result = checker.get_most_frequent(dist, n=100)
    with open('debug/dist_100.pkl', 'wb') as file:
        pickle.dump(result, file)
    print(result)