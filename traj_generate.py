import networkx as nx
import numpy as np

def generate_traj(num_traj: int, max_length: int, adj_path: str):
    G = nx.from_numpy_array(np.load(adj_path))
    trajs = []
    ods = []
    for _ in range(num_traj):
        # generate one traj
        while True:
            source = np.random.randint(1, G.number_of_nodes())
            target = np.random.randint(1, G.number_of_nodes())
            try:
                traj = nx.shortest_path(G, source=source, target=target, weight='weight')  # no random, weight para needs to debug later
                if len(traj) + 1 > max_length or len(traj) < 2:
                    continue # regenerate this one
                else:
                    break
            except nx.NetworkXNoPath:
                continue # regenrate this one
        ods.append([source, target])
        trajs.append(traj)
    ods = np.array(ods)
    padded_trajs = np.array([np.pad(traj, (0, max_length - len(traj)), 'constant', constant_values=0) for traj in trajs])
    padded_trajs = np.hstack((ods, padded_trajs))
    return padded_trajs

if __name__ == "__main__":
    ret = generate_traj(1000, 30, 'data/adj.npy')
    print(ret)
    print(ret.shape)
    np.save('data/trajs.npy', ret)