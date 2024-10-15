import pandas as pd

if __name__ == "__main__":
    node_shen_path = 'node_shenzhen.csv'
    df = pd.read_csv(node_shen_path, index_col='NodeID')
    lon_min = df['Longitude'].min() - 1e-3
    lon_max = df['Longitude'].max() + 1e-3
    lat_min = df['Latitude'].min() - 1e-3
    lat_max = df['Latitude'].max() + 1e-3
    
    print(lon_min, lon_max, lat_min, lat_max)

    # Then, Run 