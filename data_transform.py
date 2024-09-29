from lxml import etree
import numpy as np
import csv

# fetch roadnet data from osm.net.xml file
def fetch_data_from_xml(xml_file, target_dir):
    tree = etree.parse(xml_file)
    root = tree.getroot()
    
    edges = root.xpath('//edge')
    edge2attr = {}
    for edge in edges:
        id = edge.get('id')
        shape = edge.xpath('./lane/@shape')[0]
        lengths = [float(i) for i in edge.xpath('./lane/@length')]
        length = str(np.average(lengths))
        from_node_id = edge.get('from')
        to_node_id = edge.get('to')
        edge2attr[id] = {
            'shape': shape,
            'length': length
        }
    return edge2attr

if __name__ == "__main__":
    ret = fetch_data_from_xml('data/osm.net.xml', 'data')
    print(ret)