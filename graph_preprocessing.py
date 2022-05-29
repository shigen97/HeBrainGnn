import dgl
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy import sparse
from dgl.data.utils import save_graphs, load_graphs
import torch.nn.functional as F
from fmri_utils import mat2vec, vec2mat
import random
from sklearn.preprocessing import scale
from dgl import from_scipy


class BatchGraphDataSet(object):
    """
    :param graphs is a list. type of the element is DGLGraph
    :param labels is a list. type of the element is int or float
    """

    def __init__(self, graphs, labels):
        super(BatchGraphDataSet, self).__init__()
        self.graphs = graphs
        self.labels = labels

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    def __getitem__(self, idx):
        """Get the i^th sample.

        Paramters
        ---------
        idx : int
            The sample index.

        Returns
        -------
        (dgl.DGLGraph, int)
            The graph and its label.
        """
        return self.graphs[idx], self.labels[idx]


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    bathched_graph = dgl.batch(graphs)
    return bathched_graph, torch.stack(labels)


def get_dataloader(g_list, labels, batch_size=32, shuffle=True):

    dataset = BatchGraphDataSet(g_list, labels)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            collate_fn=collate)
    return dataloader


def connection2heteroGraph(fc, sc, Lobes=None):
    adj = sc
    feat = fc

    l, r = adj[::2, ::2], adj[1::2, 1::2]
    lr, rl = adj[::2, 1::2], adj[1::2, ::2]

    all_edges = [l, r, lr, rl]

    l_feat, r_feat = torch.from_numpy(feat[::2, :]).float(), torch.from_numpy(feat[1::2, :]).float()
    # l_feat = F.normalize(l_feat, dim=1, p=2)
    # r_feat = F.normalize(r_feat, dim=1, p=2)

    edge_types = [('l', 'l-l', 'l'), ('r', 'r-r', 'r'), ('l', 'l-r', 'r'), ('r', 'r-l', 'l')]
    connections = [(np.where(l != 0)), (np.where(r != 0)), (np.where(lr != 0)), (np.where(rl != 0))]
    edge_weight = [torch.from_numpy(edges[src, dst]).float() for ((src, dst), edges) in zip(connections, all_edges)]

    hetero_connections = {
        edge_type:connection for (edge_type, connection) in zip(edge_types, connections)
    }
    hetero_g = dgl.heterograph(hetero_connections)

    hetero_g.nodes['l'].data['feat'] = l_feat
    hetero_g.nodes['r'].data['feat'] = r_feat

    if Lobes is not None:
        hetero_g.nodes['l'].data['lobe'] = Lobes
        hetero_g.nodes['r'].data['lobe'] = Lobes

    hetero_g.edges['l-l'].data['w'] = edge_weight[0]
    hetero_g.edges['r-r'].data['w'] = edge_weight[1]
    hetero_g.edges['l-r'].data['w'] = edge_weight[2]
    hetero_g.edges['r-l'].data['w'] = edge_weight[3]

    return hetero_g


def load_dataloader(train, test, bacth_size=64, val=None):
    FCs = pickle.load(open('FC.pkl', 'rb'))
    SCs = pickle.load(open('SC.pkl', 'rb'))
    SCs = np.log(SCs + 1)
    SCs = SCs / np.max(SCs, (1, 2)).reshape(-1, 1, 1)
    labels = pickle.load(open('labels.pkl', 'rb'))[idx]
    labels[labels != 0] = 1

    x_FC = mat2vec(FCs)
    feat_idx = feature_selection(x_FC, labels, train, 22000)
    x_FC[:, feat_idx] = 0
    FCs = vec2mat(x_FC)


    train_FCs, train_SCs = FCs[train], SCs[train]
    test_FCs, test_SCs = FCs[test], SCs[test]

    labels = torch.from_numpy(labels).long()

    train_g_list = [connection2heteroGraph(fc, sc, None)
                    for (fc, sc) in zip(train_FCs, train_SCs)]

    test_g_list = [connection2heteroGraph(fc, sc, None)
                   for (fc, sc) in zip(test_FCs, test_SCs)]

    train_loader = get_dataloader(train_g_list, labels=labels[train], batch_size=bacth_size)
    test_loader = get_dataloader(test_g_list, labels=labels[test], batch_size=32, shuffle=False)

    return train_loader, test_loader



