from matplotlib import pyplot as plt
import csv
import numpy as np
from lxml import etree
import ast
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
    with open(traj_file, 'r') as trajs:
        # for traj in list(trajs)[:2]:
        while True:
            traj = list(trajs)[1]
            id_seq = [ast.literal_eval(traj_point)[1] for traj_point in traj.split()[1:]]
            id_seq = [id_seq[i]+1 for i in range(len(id_seq)) if i == 0 or id_seq[i] != id_seq[i - 1]]
            print(id_seq)
            loc_seq = [edges_loc.get(id, -9999) for id in id_seq]
            print(loc_seq)
            for loc in loc_seq:
                if loc != -9999:
                    plt.plot(loc[0], loc[1], linewidth=3, color='red', zorder=100)
            break
    # plt.plot(edges_loc['1'][0], edges_loc['1'][1], color='red')
    plt.savefig(fig_save_path)

if __name__ == '__main__':
    # plot_roadnet('data/edge.csv')
    # plot_roadnet_with_eccentric_road('data/edge.csv', short_threshold=0.5)
    # plot_road_legnth_hist('data/edge.csv')
    roadnet_fig, edges_loc = plot_roadnet_from_xml('data/shenzhen_8_6/edge_sumo.edg.xml', 'data/shenzhen_8_6/node_sumo.nod.xml', 'fig/shenzhen_8_6/roadnet.png')
    plot_traj(roadnet_fig, edges_loc, 'data/shenzhen_8_6/shenzhen_test_traj.txt', 'fig/shenzhen_8_6/traj.png')
    pass