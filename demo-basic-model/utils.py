import torch
import torch.nn.functional as F
from torch import Tensor, matmul
from torch_geometric.nn import GCNConv, GATv2Conv, SAGEConv, MessagePassing
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from torch_sparse import SparseTensor


class SuperLameGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='mean'):
        super().__init__(aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.liner = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.propagate(x=x, edge_index=edge_index)

    def message(self, x_j):
        return self.liner(x_j)

    def update(self, aggr_out):
        return aggr_out

    def message_and_aggregate(self, adj_t: SparseTensor) -> Tensor:
        pass

    def edge_update(self) -> Tensor:
        pass


class SuperLameGNN(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.gcn1 = SuperLameGNNLayer(dim_in, dim_h)
        self.gcn2 = SuperLameGNNLayer(dim_h, dim_out)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=0.01,
                                          weight_decay=5e-4)

    def forward(self, x, edge_index):
        h = F.dropout(x, p=0.5, training=self.training)
        h = self.gcn1(h, edge_index)
        h = torch.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.gcn2(h, edge_index)
        return h, F.log_softmax(h, dim=1)


class LameGCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, out_dim)

    def forward(self, data):
        x = self.conv1(data.x, data.adj_t)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)

        x = self.conv2(x, data.adj_t)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)

        x = self.conv3(x, data.adj_t)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)
        return x, F.log_softmax(x, dim=-1)


class LameSAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, out_dim)

    def forward(self, data):
        x = self.conv1(data.x, data.adj_t)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)

        x = self.conv2(x, data.adj_t)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)

        x = self.conv3(x, data.adj_t)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)
        return x, F.log_softmax(x, dim=-1)


class LameGAT(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2, heads=8):
        super().__init__()
        self.dropout = dropout
        self.gat1 = GATv2Conv(in_dim, hidden_dim, heads=heads)
        self.gat2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads)
        # self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.gat3 = GATv2Conv(hidden_dim * heads, out_dim, heads=1)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=0.01,
                                          weight_decay=5e-4)

    def forward(self, data):
        x = self.conv1(data.x, data.adj_t)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)

        x = self.conv3(x, data.adj_t)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)
        return x, F.log_softmax(x, dim=-1)


class EvenLamerGCN(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.gcn1 = GCNConv(dim_in, dim_h)
        self.gcn2 = GCNConv(dim_h, dim_out)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=0.01,
                                          weight_decay=5e-4)

    def forward(self, x, edge_index):
        #h = F.dropout(x, p=0.5, training=self.training)
        h = self.gcn1(x, edge_index)
        h = torch.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)

        h = self.gcn2(h, edge_index)
        return h, F.log_softmax(h, dim=1)

class EvenLamerSAGE(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.gcn1 = SAGEConv(dim_in, dim_h)
        self.gcn2 = SAGEConv(dim_h, dim_out)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=0.01,
                                          weight_decay=5e-4)

    def forward(self, x, edge_index):
        #h = F.dropout(x, p=0.5, training=self.training)
        h = self.gcn1(x, edge_index)
        h = torch.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.gcn2(h, edge_index)
        return h, F.log_softmax(h, dim=1)

class EvenLamerGAT(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, heads=8):
        super().__init__()
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
        self.gat2 = GATv2Conv(dim_h * heads, dim_out, heads=1)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=0.01,
                                          weight_decay=5e-4)

    def forward(self, x, edge_index):
        #h = F.dropout(x, p=0.5, training=self.training)
        h = self.gat1(x, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.gat2(h, edge_index)
        return h, F.log_softmax(h, dim=1)


def accuracy(pred_y, y):
    return ((pred_y == y).sum() / len(y)).item()


def train(model, data):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = model.optimizer
    epochs = 200
    result = None
    model.train()
    for epoch in range(epochs + 1):
        # Training
        optimizer.zero_grad()

        result, out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
        val_acc = accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask])

        if epoch % 10 == 0:
            print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: '
                  f'{acc * 100:>6.2f}% | Val Loss: {val_loss:.2f} | '
                  f'Val Acc: {val_acc * 100:.2f}%')

    return result


def test(model, data):
    """Evaluate the model on test set and print the accuracy score."""
    model.eval()
    _, out = model(data.x, data.edge_index)
    acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
    return acc


def scatter_embeddings(x, colors, word=False):
    label = ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 'Probabilistic_Methods', 'Reinforcement_Learning',
             'Rule_Learning', 'Theory']
    board = []
    for idx in range(10):
        board.append([])
        for j_dx in range(2):
            board[idx].append(0)
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = plt.scatter(x[:, 0], x[:, 1], lw=0, s=80, c=palette[colors.astype(int)], facecolors='none', edgecolors='r')
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        if word:
            txt = ax.text(xtext, ytext, label[i], fontsize=24)
        else:
            txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
        board[i][0] = xtext
        board[i][1] = ytext
    plt.show()
