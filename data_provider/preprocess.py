import dask.bag as db
import re
import time
import pickle

def read_shenzhen_txt_db(filename):
    bag = db.read_text(filename, blocksize='32MB')
    print(bag.npartitions)
    bag = bag.map(lambda line: [int(i) for i in re.findall(r'\d+', line)[1:]])
    results = bag.map(lambda traj: [(traj[i], traj[i+1]) for i in range(0, len(traj), 2)]).compute()
    return results

def shenzhen_txt_to_pkl():
    for data_type in ['train', 'val', 'test']:
        data_type = 'test'
        file_path = f"data_provider/data/shenzhen_8_6/shenzhen_{data_type}_traj.txt"
        start_time  = time.time()
        results = read_shenzhen_txt_db(file_path)
        print("Time: ", time.time() - start_time)

        save_path = f'data_provider/data/shenzhen_8_6/shenzhen_{data_type}_traj.pkl'
        with open(save_path, 'wb') as file:
            pickle.dump(results, file) 

if __name__ == "__main__":
    data_type = 'test'
    file_path = f"data_provider/data/shenzhen_8_6/shenzhen_{data_type}_traj.txt"
    start_time  = time.time()
    results = read_shenzhen_txt_db(file_path)
    print("Time: ", time.time() - start_time)

    save_path = f'data_provider/data/shenzhen_8_6/shenzhen_{data_type}_traj.pkl'
    with open(save_path, 'wb') as file:
        pickle.dump(results, file) 