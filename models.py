from dgl.nn.pytorch import GraphConv, SGConv
import torch as th
from torch import nn
from torch.nn import init
from dgl import laplacian_lambda_max, broadcast_nodes, function as fn
import torch.nn.functional as F
import dgl
import numpy as np
import random

class SGCLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 k=1,
                 cached=False,
                 bias=True,
                 norm=None,
                 allow_zero_in_degree=True):
        super(SGCLayer, self).__init__()
        self.gcn_layer = SGConv(in_feats,
                                out_feats,
                                k,
                                cached,
                                bias,
                                norm,
                                allow_zero_in_degree)
        self.res = nn.Linear(in_feats, out_feats)

    def forward(self, bg, feat):
        return self.gcn_layer(bg, feat) + self.res(feat)


class InnerProductDecoder(nn.Module):
    def __init__(self, activation=th.sigmoid, dropout=0.0):
        super(InnerProductDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, z1, z2):
        z1 = self.dropout(z1)
        z2 = self.dropout(z2)
        adj = th.mm(z1, z2.t())
        if self.activation is not None:
            adj = self.activation(adj)
        return adj


class MLPDecoder(nn.Module):
    def __init__(self, hidden_dim, activation=th.sigmoid, dropout=0.3):
        super(MLPDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.edge_layer_l = nn.Linear(hidden_dim, 1)
        self.edge_layer_r = nn.Linear(hidden_dim, 1)

    def forward(self, z1, z2):
        z1 = self.dropout(z1)
        z2 = self.dropout(z2)
        l, r = self.edge_layer_l(z1), self.edge_layer_r(z2)
        adj = l + r.T
        if self.activation is not None:
            adj = self.activation(adj)
        return adj


class GraphConvsg(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm=True,
                 bias=True,
                 weighted=False,
                 activation=None):
        super(GraphConvsg, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._weighted = weighted

        self.send = fn.copy_src(src='h', out='m')
        self.recv = fn.sum(msg='m', out='h')
        if self._weighted:
            self.send = fn.u_mul_e('h', 'w', 'm')
        self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, graph, feat):
        graph = graph.local_var()
        if self._norm:
            try:
                norm = th.pow(graph.ndata['in_degrees'].float().clamp(min=1), -0.5)
            except KeyError:
                norm = th.pow(graph.in_degrees().float().clamp(min=1), -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = th.reshape(norm, shp).to(feat.device)
            feat = feat * norm

        if self._in_feats > self._out_feats:
            # mult W first to reduce the feature size for aggregation.
            feat = th.matmul(feat, self.weight)
            graph.ndata['h'] = feat
            graph.update_all(self.send, self.recv)
            rst = graph.ndata['h']
        else:
            # aggregate first then mult W
            graph.ndata['h'] = feat
            graph.update_all(self.send, self.recv)
            rst = graph.ndata['h']
            rst = th.matmul(rst, self.weight)

        if self._norm:
            rst = rst * norm

        if self.bias is not None:
            rst = rst + self.bias

        if self._activation is not None:
            rst = self._activation(rst)

        return rst

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation=F.relu, weighted=False, bias=True,
                 norm=True, residual=True, batchnorm=True, dropout=0.):
        super(GCNLayer, self).__init__()

        self.activation = activation
        self.graph_conv = GraphConvsg(in_feats=in_feats, out_feats=out_feats, bias=bias,
                                      norm=norm, weighted=weighted, activation=activation)
        self.dropout = nn.Dropout(dropout)

        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(in_feats, out_feats)

        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)

    def forward(self, bg, feats):
        new_feats = self.graph_conv(bg, feats)

        if self.residual:
            res_feats = self.res_connection(feats)
            new_feats = new_feats + res_feats
        new_feats = self.dropout(new_feats)

        if self.bn:
            new_feats = self.bn_layer(new_feats)

        return new_feats



class HeteroGraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 bias=False,
                 weighted=False,
                 residual=False,
                 activation=F.relu,
                 dropout=0.0):
        super(HeteroGraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._weighted = weighted

        self.send = fn.copy_src(src='h', out='m')
        self.recv = fn.mean(msg='m', out='h')
        self.edge_func = nn.Linear(1, out_feats)
        if self._weighted:
            self.send = fn.u_mul_e('h', 'w_feat', 'm')

        self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.activation = activation
        self.bn = nn.BatchNorm1d(out_feats)

        self.dropout = nn.Dropout(dropout)

        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(in_feats, out_feats)

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, graph, input_feat, etype):
        graph = graph.local_var()
        graph.edges[etype].data['w_feat'] = self.edge_func(graph.edges[etype].data['w'].reshape(-1, 1))

        src_type = etype[:etype.index('-')]
        dst_type = etype[etype.index('-') + 1:]

        # mult W first to reduce the feature size for aggregation.
        feat = th.matmul(input_feat, self.weight)
        graph.nodes[src_type].data['h'] = feat
        graph.update_all(self.send, self.recv, etype=etype)
        rst = graph.nodes[dst_type].data['h']


        if self.bias is not None:
            rst = rst + self.bias

        if self.activation is not None:
            rst = self.activation(rst)
            
        if self.residual:
            res_feats = self.res_connection(input_feat)
            if self.activation is not None:
                res_feats = self.activation(res_feats)
            rst = rst + res_feats
            
        new_feats = self.dropout(rst)
        return new_feats

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)


class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, etypes, activation=F.relu, weighted=False, bias=False,
                 residual=False, dropout=0., mode='hetero'):
        super(HeteroRGCNLayer, self).__init__()
        self.mode = mode
        if self.mode == 'hetero':
            self.gcn_layers = nn.ModuleDict({
                    etype: HeteroGraphConv(in_feats=in_feats, out_feats=out_feats, activation=None,
                                           weighted=weighted, bias=bias, residual=False,
                                           dropout=dropout) for etype in etypes
                })
        else:
            assert self.mode == 'homo'
            conv = HeteroGraphConv(in_feats=in_feats, out_feats=out_feats, activation=None,
                                           weighted=weighted, bias=bias, residual=False,
                                           dropout=dropout)
            self.gcn_layers = nn.ModuleDict({
                'l-r': conv,
                'l-l': conv
            })
        self.activation = activation

        self.etypes = etypes
        self.residual = residual
        if self.residual:
            self.self_layer = nn.Linear(in_feats, out_feats)

    def forward(self, G, feat_dict, R_aggregate_mode='mean'):
        G = G.local_var()
        for etype in self.etypes:
            if etype == 'l-r':
                for (srctype, dsttype) in [('l', 'r'), ('r', 'l')]:

                    etype_gcn_layer = self.gcn_layers[etype]
                    src_feat = feat_dict[srctype]
                    # dst_etype_feat = etype_gcn_layer(G, src_feat, etype)
                    real_etype = srctype + '-' + dsttype
                    dst_etype_feat = etype_gcn_layer(G, src_feat, real_etype)
                    G.nodes[dsttype].data['inter'] = dst_etype_feat
            else:
                for (srctype, dsttype) in [('l', 'l'), ('r', 'r')]:
                    etype_gcn_layer = self.gcn_layers[etype]
                    src_feat = feat_dict[srctype]
                    # dst_etype_feat = etype_gcn_layer(G, src_feat, etype)
                    real_etype = srctype + '-' + dsttype
                    dst_etype_feat = etype_gcn_layer(G, src_feat, real_etype)
                    G.nodes[dsttype].data['intra'] = dst_etype_feat


        new_feature_dict = {}
        for ntype in G.ntypes:
            if G.num_nodes(ntype) != 0:
                ntype_feats = [G.nodes[ntype].data['inter'], G.nodes[ntype].data['intra']]
                # if self.residual:
                #     ntype_feats.append(self.self_layer(feat_dict[ntype]))
                ntype_feat = th.mean(th.stack(ntype_feats), 0)
                if self.activation is not None:
                    ntype_feat = self.activation(ntype_feat)
                if self.residual:
                    ntype_feat += self.self_layer(feat_dict[ntype])
                new_feature_dict[ntype] = ntype_feat

        return new_feature_dict


class Discriminator(nn.Module):
    def __init__(self, n_h1, n_h2):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h1, n_h2, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            th.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, anchor_feat, pos_feat, neg_feat):

        sc_pos = th.sigmoid(th.squeeze(self.f_k(pos_feat, anchor_feat)))
        sc_neg = th.sigmoid(th.squeeze(self.f_k(neg_feat, anchor_feat)))

        return sc_pos, sc_neg


class HeteroGraphCLF(nn.Module):
    def __init__(self, in_dim, hidden_dims, etypes, dropout=0.0, activation=F.relu,
                 weighted=False, bias=False, residual=True, meta_paths=None, mode='hetero'):
        super(HeteroGraphCLF, self).__init__()
        self.mode = mode
        self.etypes = etypes
        self.meta_paths = meta_paths
        self.ntypes = ['l', 'r']
        layers = [HeteroRGCNLayer(in_feats=in_dim, out_feats=hidden_dims[0], etypes=etypes,
                                  activation=activation, weighted=weighted, bias=bias,
                                  residual=residual, dropout=dropout, mode=mode)]
        if len(hidden_dims) >= 2:
            for i in range(1, len(hidden_dims)):
                layers.append(HeteroRGCNLayer(in_feats=hidden_dims[i - 1], out_feats=hidden_dims[i], etypes=etypes,
                                              activation=activation, weighted=weighted, bias=bias,
                                              residual=residual, dropout=dropout, mode=mode))

        self.encoder = nn.ModuleList(layers)

        self.classify_layer = nn.Sequential(nn.Linear(246*2, 2))
        self.readout_layer = nn.Linear(hidden_dims[-1], 1)
        self.ntypes_meta_path_emb = nn.ModuleDict({})
        self.scores_func = nn.ModuleDict({})
        for step, ntype in enumerate(self.ntypes):
            self.ntypes_meta_path_emb[ntype] = nn.ModuleDict({
                meta_path_name: SGCLayer(in_feats=in_dim,
                                         out_feats=hidden_dims[0],
                                         k=2,
                                         cached=False,
                                         bias=False,
                                         norm=None,
                                         allow_zero_in_degree=True)


                for meta_path_name, _ in self.meta_paths[ntype]})

            self.scores_func[ntype] = Discriminator(hidden_dims[-1], hidden_dims[-1])

    def forward(self, g, k=3, method='mean'):
        feat_dict = g.ndata['feat']
        batch_size = g.batch_size
        for conv in self.encoder:
            feat_dict = conv(g, feat_dict)


        g.ndata['h'] = feat_dict


        pos_scores, neg_scores = [], []
        for time in range(k):
            for step, ntype in enumerate(self.ntypes):
                h_anchor = feat_dict[ntype]
                bg_ntype_meta_graphs = {meta_path_name: dgl.metapath_reachable_graph(g, meta_path)
                    .remove_self_loop().add_self_loop().to(g.device)
                                        for meta_path_name, meta_path in self.meta_paths[ntype]}

                h_pos = [
                    self.ntypes_meta_path_emb[ntype][meta_path_name](bg_ntype_meta_graphs[meta_path_name],
                                                                     g.ndata['feat'][ntype])
                    for meta_path_name, _ in self.meta_paths[ntype]]
                if time == 0:
                    g.nodes[ntype].data['mp_feat'] = h_pos[0]

                all_idx = np.arange(h_anchor.shape[0]).tolist()
                bnn = g.batch_num_nodes(ntype)
                for i in range(bnn.shape[0]):
                    batch_idx = all_idx[bnn[:i].sum().item(): bnn[:i + 1].sum().item()]
                    random.shuffle(batch_idx)
                    all_idx[bnn[:i].sum().item(): bnn[:i + 1].sum().item()] = batch_idx

                h_neg = [h[all_idx] for h in h_pos]

                for i in range(len(h_pos)):
                    pos_score, neg_score = self.scores_func[ntype](h_anchor, h_pos[i], h_neg[i])
                    pos_scores.append(pos_score)
                    neg_scores.append(neg_score)

        pos_scores, neg_scores = th.cat(pos_scores), th.cat(neg_scores)

        feat = self.readout(g, method=method)

        prediction = self.classify_layer(feat)

        return prediction, pos_scores, neg_scores
