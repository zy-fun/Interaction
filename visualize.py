from matplotlib import pyplot as plt
import csv
import numpy as np
from lxml import etree
import ast
import pickle
from pyspark.sql import SparkSession
from tqdm import tqdm
from matplotlib.collections import LineCollection
from pyspark.sql.functions import col, sum
# import pandas as pd

def plot_roadnet(edge_file):
    with open(edge_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            shape = [[float(i) for i in coord.split(',')] for coord in row['shape'].split(' ')]
            x, y = list(zip(*shape))
            plt.plot(x, y, color='black')
    plt.savefig('roadnet.png')

def plot_roadnet_with_eccentric_road(edge_file, short_threshold=0.75, long_threshold=2):
    with open(edge_file, 'r') as f:
        reader = csv.DictReader(f)
        lengths = []
        for row in reader:
            lengths.append(float(row['length']))
        lengths = np.array(lengths)
        avg, std = np.mean(lengths), np.std(lengths)
        print('avg: ', avg)
        print('std: ', std)

    with open(edge_file, 'r') as f:
        reader = csv.DictReader(f)
        overlong_count = 0
        overshort_count = 0
        for row in reader:
            shape = [[float(i) for i in coord.split(',')] for coord in row['shape'].split(' ')]
            x, y = list(zip(*shape))
            if float(row['length']) > avg + long_threshold * std:
                plt.plot(x, y, color='red', zorder=10)
                overlong_count += 1
            elif float(row['length']) < avg - short_threshold * std:
                plt.plot(x, y, color='blue', zorder=11)
                overshort_count += 1
            else:
                plt.plot(x, y, color='black', alpha=0.2)
                pass
    
        plt.plot([], [], color='red', label=f"overlong: {overlong_count}")
        plt.plot([], [], color='blue', label=f"overshort: {overshort_count}")
        plt.legend()
    plt.savefig('roadnet.png')

def plot_road_legnth_hist(edge_file):
    with open(edge_file, 'r') as f:
        reader = csv.DictReader(f)
        lengths = []
        for row in reader:
            lengths.append(float(row['length']))
    plt.hist(lengths, bins=100)
    plt.savefig('road_length_hist.png')

def plot_roadnet_from_xml(edges_file, nodes_file, fig_save_path, mapping_dict=None):
    nodes_tree = etree.parse(nodes_file)
    nodes = nodes_tree.getroot()
    node_loc = {
        node.get('id') : (float(node.get('x')), float(node.get('y'))) 
            for node in nodes.xpath('./node')}

    plt.figure(figsize=(16, 16))
    edges_tree = etree.parse(edges_file)
    edges = edges_tree.getroot()
    edges_loc = {}
    for edge in edges.xpath('./edge'):
        from_id = edge.get('from')
        to_id = edge.get('to')
        x, y = list(zip(node_loc[from_id], node_loc[to_id]))
        # index = mapping_dict[edge.get('id')] if mapping_dict else int(edge.get('id'))
        index = int(edge.get('id'))
        edges_loc[index] = x, y
        plt.plot(x, y, color='black', alpha=.1)
        
    plt.savefig(fig_save_path)
    return plt.gcf(), edges_loc

# def get_mapping_dict_from_csv(edge_mapping_file):
#     '''
#         mapping 'Edge ID' (which is used in edge_sumo.edg.xml)
#         to 'Index' (which is used in traj files. e.g. shenzhen_test_traj.txt)
#     '''
#     import pandas as pd
#     df = pd.read_csv(edge_mapping_file)
#     mapping_dict = dict(zip(df['Edge ID'], df['Index']))
#     return mapping_dict

def plot_traj(fig, edges_loc, traj_file, fig_save_path):
    plt.figure(fig)
    with open(traj_file, 'r') as file:
        for traj in file: 
            id_seq = [ast.literal_eval(traj_point)[1] for traj_point in traj.split()[1:]]
            id_seq = [id_seq[i]+1 for i in range(len(id_seq)) if i == 0 or id_seq[i] != id_seq[i - 1]]
            loc_seq = [edges_loc.get(id, -9999) for id in id_seq]
            for loc in loc_seq:
                if loc != -9999:
                    plt.plot(loc[0], loc[1], linewidth=3, color='red', zorder=100)
    plt.savefig(fig_save_path)

def plot_loss(loss_file, fig_save_path, slice_length=140):
    with open(loss_file, 'rb') as f:
        loss = pickle.load(f)

    loss = [sum(loss[i * slice_length: (i+1) *slice_length]) / slice_length for i in range(0, len(loss) // slice_length)]
    batches = list(range(len(loss)))
    print(loss)
    plt.plot(batches, loss, scaley=False)
    plt.savefig(fig_save_path)

class Shenzhen_Visualizer:
    def __init__(self):
        self.spark = SparkSession.builder \
            .config("spark.executor.memory", "8g") \
            .config("spark.driver.memory", "8g") \
            .config("spark.memory.offHeap.enabled", "true") \
            .config("spark.memory.offHeap.size", "4g") \
            .master("local") \
            .appName("get_num_of_data_points"). \
            getOrCreate()

        self.density_distribution_visualize()
        # self.direction_field_visualize(data_type='train', traj_indices=list(np.random.randint(0, 1000, 10)))
        # self.intersection_visualize()
        # for edge_id in tqdm(np.random.randint(0, 26200, 100)):
        #     self.edge_density_visualize(edge_id=edge_id)

    def density_distribution_visualize(self):
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
        density_path = 'debug/shenzhen_8_6_density_by_lane.parquet'
        df = self.spark.read.parquet(density_path)
        
        for direction in direction_dict:
            direction_df = df.filter(col('direction') == direction).drop('direction', 'time') \
                .groupby('edge_id') \
                .agg(sum('density').alias('density'))
            direction_density = direction_df.agg(sum('density')).collect()[0][0]
            print(direction_dict[direction], direction_density)
            pdf = direction_df.toPandas()
            plt.figure(figsize=(12, 6))
            plt.bar(pdf['edge_id'], pdf['density'], color=color_dict[direction], label=direction_dict[direction])
            plt.title(f'Density Distribution by Edge ({direction_dict[direction]})')
            plt.xlabel('Edge ID')
            plt.ylabel('Density')
            plt.legend()
            plt.savefig(f'fig/shenzhen_8_6/density_distribution_by_edge_{direction}.png')

    def edge_density_visualize(self, edge_id):
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

        density_path = 'debug/shenzhen_8_6_density_by_lane.parquet'
        df = self.spark.read.parquet(density_path)
        df = df.filter(col('edge_id') == edge_id).drop('edge_id').orderBy('time')

        plt.figure(figsize=(12, 6))
        for direction in direction_dict:
            direction_df = df.filter(col('direction') == direction).drop('direction')
            pdf = direction_df.toPandas()
        
            plt.plot(pdf['time'], 
                    pdf['density'], 
                    color=color_dict[direction],
                    label=direction_dict[direction])

        plt.title(f'Density over Time on Edge {edge_id}')
        plt.xlabel('Time')
        plt.ylabel('Density')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()

        plt.savefig(f'fig/shenzhen_8_6/density_by_lane_visualize/density_time_series_{edge_id}.png', dpi=300, bbox_inches='tight')
        plt.close()


    def intersection_visualize(self):
        nodes_path = "data_provider/data/shenzhen_8_6/nodes_with_intersection.parquet"
        edges_path = "data_provider/data/shenzhen_8_6/edges_with_direction.parquet"

        nodes_df = self.spark.read.parquet(nodes_path)
        edges_df = self.spark.read.parquet(edges_path)

        segments = edges_df.select('from_x', 'from_y', 'to_x', 'to_y').toPandas().values
        segments = segments.reshape(-1, 2, 2)
        fig, ax = plt.subplots(figsize=(12, 12))

        lc = LineCollection(segments, 
                        colors='blue',
                        linewidths=0.5,
                        alpha=0.6)        
        ax.add_collection(lc)
        ax.autoscale() 

        intersection_coords = nodes_df.filter(col('intersection or dead end') == 1).select('x', 'y').toPandas().values
        sc = ax.scatter(intersection_coords[:, 0],  # x坐标
                    intersection_coords[:, 1],    # y坐标
                    c='red',               # 颜色
                    s=5,                  # 点的大小
                    alpha=0.6,             # 透明度
                    marker='o')            # 点的形状

        non_intersection_coords = nodes_df.filter(col('intersection or dead end') == 0).select('x', 'y').toPandas().values
        sc = ax.scatter(non_intersection_coords[:, 0],  # x坐标
                    non_intersection_coords[:, 1],    # y坐标
                    c='green',               # 颜色
                    s=3,                  # 点的大小
                    alpha=0.3,             # 透明度
                    marker='o')            # 点的形状

        plt.title('Road Network with Intersections')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.savefig('fig/shenzhen_8_6/roadnet_with_intersections.png')
        plt.close()

    def direction_field_visualize(self, data_type='test', traj_indices=[]):
        traj_path = f"data_provider/data/shenzhen_8_6/shenzhen_{data_type}_traj.parquet"
        edge_path = f"data_provider/data/shenzhen_8_6/edges_with_direction.parquet"

        traj_df = self.spark.read.parquet(traj_path)
        edge_df = self.spark.read.parquet(edge_path)
        traj_df = traj_df.filter(traj_df.traj_id.isin(traj_indices) & (traj_df.about_to_move == 1))
        edge_ids = set(traj_df.select('edge_id').distinct().rdd.map(lambda x: x[0]).collect())
    
        edge_dict = dict(
            edge_df.rdd
            .filter(lambda row: row.id in edge_ids)
            .map(lambda row: (row.id, ((row.from_x, row.to_x), (row.from_y, row.to_y))))
            .collect()
        )
    
        edge_dict_broadcast = self.spark.sparkContext.broadcast(edge_dict)
        
        def single_trajectory_direction_field_visualize(data_type, traj_id):
            # 0 for left, 1 for straight, 2 for right, 3 for back
            # traj_copy = self.spark.createDataFrame(traj_df.toPandas())
            traj = traj_df.filter(traj_df.traj_id == traj_id).collect()
            
            color_dict = {
                0: 'red',
                1: 'blue',
                2: 'green',
                3: 'yellow',
            }
            label_dict = {
                0: 'Left',
                1: 'Straight',
                2: 'Right',
                3: 'Back'
            }
        
            for i, point in enumerate(traj):
                edge_id = point.edge_id
                x_coord, y_coord = edge_dict[edge_id]
                if i == 0:
                    plt.plot(x_coord[0], y_coord[0], color='black', label='Start', marker='o', ms=10)
                elif i == len(traj) - 1:
                    plt.plot(x_coord[-1], y_coord[-1], color='black', label='End', marker='*', ms=15)

                plt.plot(x_coord, y_coord, color=color_dict[point.direction], label='', marker='o', ms=5)

            plt.title(f'Direction Field Visualization of Trajectory {traj_id} in {data_type} Set')
            
            handles = [plt.plot([], [], color=color_dict[i], label=label_dict[i])[0] for i in range(4)]
            plt.legend(handles=handles)
            plt.savefig(f'fig/shenzhen_8_6/direction_field_visualize/{data_type}_{traj_id}.png')
            plt.close()
        
        for traj_id in tqdm(traj_indices):
            single_trajectory_direction_field_visualize(data_type=data_type, traj_id=traj_id)
        

if __name__ == '__main__':
    # plot_roadnet('data/edge.csv')
    # plot_roadnet_with_eccentric_road('data/edge.csv', short_threshold=0.5)
    # plot_road_legnth_hist('data/edge.csv')
    # roadnet_fig, edges_loc = plot_roadnet_from_xml('data_provider/data/shenzhen_8_6/edge_sumo.edg.xml', 'data_provider/data/shenzhen_8_6/node_sumo.nod.xml', 'fig/shenzhen_8_6/roadnet.png')
    # plot_traj(roadnet_fig, edges_loc, 'data_provider/data/shenzhen_8_6/shenzhen_val_traj.txt', 'fig/shenzhen_8_6/traj_val.png')
    # plot_loss('checkpoints/241114_2058 dataname_binary_classification windowsize_2 blockdims_[64, 32, 32] trainepochs_1000 batchsize_64 learningrate_0.008/loss.pkl', 'fig/loss.png')
    Shenzhen_Visualizer()
    pass