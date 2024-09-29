from lxml import etree
import numpy as np
import csv
import os

# fetch roadnet data from osm.net.xml file
def fetch_data_from_xml(xml_file, target_dir):
    # create edge_dict from xml file
    tree = etree.parse(xml_file)
    root = tree.getroot()
    
    # edges = root.xpath('//edge')
    edges = root.xpath('//edge[@from and @to]')
    edge_dict = {}
    for edge in edges:
        edge_id = edge.get('id')
        shape = edge.xpath('./lane/@shape')[0]
        lengths = [float(i) for i in edge.xpath('./lane/@length')]
        length = str(np.average(lengths))
        from_node_id = edge.get('from')
        to_node_id = edge.get('to')
        edge_dict[edge_id] = {
            'edge_id': edge_id,
            'from_node_id': from_node_id,
            'to_node_id': to_node_id,
            'shape': shape,
            'length': length
        }

    # edges = root.xpath('//edge[@from and @to]')
    # for node in nodes:

    
    # write edge_dict to csv file
    csv_file = os.path.join(target_dir, 'edge.csv')
    with open(csv_file, 'w') as f:
        field_names = ['edge_id', 'from_node_id', 'to_node_id', 'shape', 'length']
        dict_writer = csv.DictWriter(f, field_names)
        dict_writer.writeheader()
        for _, edge in edge_dict.items():
            dict_writer.writerow(edge)
    
    # construct adj

if __name__ == "__main__":
    ret = fetch_data_from_xml('data/osm.net.xml', 'data')
    print(ret)