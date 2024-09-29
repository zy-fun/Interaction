from lxml import etree
import numpy as np
import csv
import os

# fetch roadnet data from osm.net.xml file
def fetch_data_from_xml(xml_file: str, target_dir: str):
    """

    """

    # create edge_dict from xml file
    tree = etree.parse(xml_file)
    root = tree.getroot()
    
    # edges = root.xpath('//edge')
    edges = root.xpath('//edge[@from and @to]')
    edge_dict = {}
    edge_id = 0
    for edge in edges:
        edge_id += 1
        osm_edge_id = edge.get('id')
        shape = edge.xpath('./lane/@shape')[0]
        lengths = [float(i) for i in edge.xpath('./lane/@length')]
        length = str(np.average(lengths))
        osm_from_node_id = edge.get('from')
        osm_to_node_id = edge.get('to')
        edge_dict[osm_edge_id] = {
            'edge_id': edge_id,
            'osm_edge_id': osm_edge_id,
            'osm_from_node_id': osm_from_node_id,
            'osm_to_node_id': osm_to_node_id,
            'shape': shape,
            'length': length
        }

    # create connection_dict from xml file
    all_connections = root.xpath('//connection')
    valid_connections = []
    for connection in all_connections:
        osm_from_edge_id = connection.get('from')
        osm_to_edge_id = connection.get('to')
        if osm_from_edge_id in edge_dict and osm_to_edge_id in edge_dict:   # this causes redundant connections
            from_edge_id = edge_dict[osm_from_edge_id]['edge_id']
            to_edge_id = edge_dict[osm_to_edge_id]['edge_id']
            valid_connections.append({
                'from_edge_id': from_edge_id,
                'to_edge_id': to_edge_id,
                'osm_from_edge_id': osm_from_edge_id,
                'osm_to_edge_id': osm_to_edge_id
            })

    # create node from xml
    # edges = root.xpath('//edge[@from and @to]')
    # for node in nodes:

    
    # write edge_dict to csv file
    csv_file = os.path.join(target_dir, 'edge.csv')
    with open(csv_file, 'w') as f:
        field_names = ['edge_id', 'osm_edge_id', 'osm_from_node_id', 'osm_to_node_id', 'shape', 'length']
        dict_writer = csv.DictWriter(f, field_names)
        dict_writer.writeheader()
        for _, edge in edge_dict.items():
            dict_writer.writerow(edge)
    
    # write valid_connections to csv file
    csv_file = os.path.join(target_dir, 'connection.csv')
    with open(csv_file, 'w') as f:
        field_names = ['from_edge_id', 'to_edge_id', 'osm_from_edge_id', 'osm_to_edge_id']
        dict_writer = csv.DictWriter(f, field_names)
        dict_writer.writeheader()
        for connection in valid_connections:
            dict_writer.writerow(connection)
            
    # construct adj
    adj = np.zeros((len(edge_dict) + 1, len(edge_dict) + 1))    # real edge_id starts from 1, 0 is remained for virtual edge
    print(len(valid_connections))
    print(valid_connections[0])
    for connection in valid_connections:
        adj[connection['from_edge_id'], connection['to_edge_id']] = 1
    np.save(os.path.join(target_dir, 'adj.npy'), adj)
    return adj

if __name__ == "__main__":
    ret = fetch_data_from_xml('data/osm.net.xml', 'data')
    print(ret)
    print(np.sum(ret))