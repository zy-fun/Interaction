from matplotlib import pyplot as plt
import csv
# import pandas as pd

def plot_roadnet(edge_file):
    with open(edge_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            shape = [[float(i) for i in coord.split(',')] for coord in row['shape'].split(' ')]
            x, y = list(zip(*shape))
            plt.plot(x, y)
    plt.savefig('roadnet.png')

if __name__ == '__main__':
    plot_roadnet('data/edge.csv')
    pass