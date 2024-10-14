import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import networkx as nx
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import csv
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# GCN模型定义
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # 第一次卷积
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # 第二次卷积
        x = self.conv2(x, edge_index)
        return x

# 创建图数据（例如使用networkx构建)
G = nx.DiGraph()

# 在我们的架构中，路段作为图结构中的节点
with open('data/edge.csv', 'r') as f:
    reader = csv.DictReader(f)
    for roadseg in reader:
        # 路段属性(长度、限速、车道数)
        roadseg_attr = {'length': float(roadseg['length'])}
        G.add_node(roadseg['osm_edge_id'], **roadseg)

# 在我们的架构中，路段之间的连接关系作为图结构中的边
with open('data/connection.csv', 'r') as f:
    reader = csv.DictReader(f)
    for connection in reader:
        G.add_edge(connection['from_osm_edge_id'], connection['to_osm_edge_id'])

# 节点特征和标签
num_features = 32
hidden_dim = 64
output_dim = 32
x = torch.rand(G.number_of_nodes(), num_features)
adj = nx.to_scipy_sparse_array(G)
edge_index = torch.tensor(np.array(adj.nonzero()), dtype=torch.long)
data = Data(x=x, edge_index=edge_index)

def contrastive_loss(z, edge_index, temperature=0.5):
    # 归一化嵌入
    z = F.normalize(z, p=2, dim=1)

    # 生成正样本：邻接矩阵中的相连点对
    pos_samples = torch.mm(z[edge_index[0]], z[edge_index[1]].T).diag()

    # 生成负样本：随机选择不相连的点对作为负样本
    num_nodes = z.size(0)
    num_neg_samples = edge_index.size(1)
    neg_edge_index = torch.randint(0, num_nodes, edge_index.size(), device=edge_index.device)

    # 负样本的相似度
    neg_samples = torch.mm(z[neg_edge_index[0]], z[neg_edge_index[1]].T).diag()

    # 拼接正负样本并计算对比损失
    labels = torch.cat([torch.ones_like(pos_samples), torch.zeros_like(neg_samples)])
    logits = torch.cat([pos_samples, neg_samples]) / temperature

    loss = F.binary_cross_entropy_with_logits(logits, labels)
    return loss

model = GCN(num_features, hidden_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

def plot_node_embeddings(node_embeddings, edge_index, filename):
    node_embeddings_np = node_embeddings.detach().cpu().numpy()
    pca = PCA(n_components=2)  # 降维到2D
    node_embeddings_2d = pca.fit_transform(node_embeddings_np)

    plt.figure(figsize=(8, 8))
    plt.scatter(node_embeddings_2d[:, 0], node_embeddings_2d[:, 1], c='blue', s=50,zorder=100)

    for i, j in edge_index.t().tolist():
        plt.plot(node_embeddings_2d[[i,j], 0], node_embeddings_2d[[i,j], 1], c='black', alpha=0.2,zorder=0)

    plt.title(filename)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.savefig(filename)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()

    # 获取节点嵌入
    z = model(data)
    plot_node_embeddings(z, data.edge_index, f'gnnfig/epoch_{epoch}.png')
    
    loss = contrastive_loss(z, data.edge_index)
    
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')


# 获取最终节点嵌入
model.eval()
with torch.no_grad():
    node_embeddings = model(data)
    print(node_embeddings[0])
    print("Final node embeddings shape:", node_embeddings.shape)

