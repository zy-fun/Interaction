from matplotlib import pyplot as plt
import csv
import numpy as np
from lxml import etree
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

def plot_roadnet_from_xml(edges_file, nodes_file, fig_save_path):
    nodes_tree = etree.parse(nodes_file)
    nodes = nodes_tree.getroot()
    node_loc = {
        node.get('id') : (float(node.get('x')), float(node.get('y'))) 
            for node in nodes.xpath('./node')}

    plt.figure(figsize=(16, 16))
    edges_tree = etree.parse(edges_file)
    edges = edges_tree.getroot()
    for edge in edges.xpath('./edge'):
        from_id = edge.get('from')
        to_id = edge.get('to')
        x, y = list(zip(node_loc[from_id], node_loc[to_id]))
        plt.plot(x, y, color='black', alpha=.5)
        
    plt.savefig(fig_save_path)


if __name__ == '__main__':
    # plot_roadnet('data/edge.csv')
    # plot_roadnet_with_eccentric_road('data/edge.csv', short_threshold=0.5)
    # plot_road_legnth_hist('data/edge.csv')
    plot_roadnet_from_xml('data/shenzhen/edge_sumo.edg.xml', 'data/shenzhen/node_sumo.nod.xml', 'fig/shenzhen/roadnet.png')
    pass