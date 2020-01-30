from __future__ import print_function
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('%s/pytorch_structure2vec-master/s2v_lib' % os.path.dirname(
    os.path.realpath(__file__)))

from torch_geometric.nn.models import GraphUNet
from torch_geometric.nn import global_sort_pool

class GUNet(nn.Module):
    def __init__(self, output_dim, num_node_feats, num_edge_feats,
                 latent_dim=[32, 32, 32, 1], k=30, conv1d_channels=[16, 32],
                 conv1d_kws=[0, 5]):
        print('Initializing GUNet')
        super(GUNet, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats
        self.k = k
        self.total_latent_dim = sum(latent_dim)
        conv1d_kws[0] = self.total_latent_dim

        self.conv_params = nn.ModuleList()
        # print(num_node_feats)
        self.conv_params.append(nn.Linear(num_node_feats, latent_dim[0]))
        for i in range(1, len(latent_dim)):
            self.conv_params.append(nn.Linear(latent_dim[i-1], latent_dim[i]))

        self.conv1d_params1 = nn.Conv1d(
            1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = nn.MaxPool1d(2, 2)
        self.conv1d_params2 = nn.Conv1d(
            conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)

        dense_dim = int((k - 2) / 2 + 1)
        self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]

        if num_edge_feats > 0:
            self.w_e2l = nn.Linear(num_edge_feats, latent_dim)
        if output_dim > 0:
            self.out_params = nn.Linear(self.dense_dim, output_dim)
        # ks = [4000, 3000, 2000, 1000]
        ks = [0.9, 0.7, 0.6, 0.5]
        output_channels_gUnet = sum(latent_dim)
        self.gUnet = GraphUNet(num_node_feats, hidden_channels=48, out_channels=output_channels_gUnet,
                               depth=len(ks), pool_ratios=ks).cuda()

    def forward(self, data):
        h = self.sortpooling_embedding_tg(data)

        return h

    def sortpooling_embedding_tg(self, data):
        ''' if exists edge feature, concatenate to node feature vector '''
        node_feat, edge_index, batch = data.x, data.edge_index, data.batch
        # TODO: remove edge_attr consideration
        # if data.edge_attr is not None:
        #     input_edge_linear = self.w_e2l(data.edge_attr)
        #     e2npool_input = gnn_spmm(e2n_sp, input_edge_linear)
        #     node_feat = torch.cat([node_feat, e2npool_input], 1)

        '''G-UNet Layer to process the graph data'''
        # the output feature dimension of gUnet is
        cur_message_layer = self.gUnet(x=node_feat, edge_index=edge_index, batch=batch)
        ''' sortpooling layer '''
        # the shape of global_sort_pool is (B, k*total_latent_dim)
        batch_sortpooling_graphs = global_sort_pool(cur_message_layer, batch, self.k)

        ''' traditional 1d convlution and dense layers '''
        to_conv1d = batch_sortpooling_graphs.view((-1, 1, self.k * self.total_latent_dim))
        conv1d_res = self.conv1d_params1(to_conv1d)
        conv1d_res = F.relu(conv1d_res)
        conv1d_res = self.maxpool1d(conv1d_res)
        conv1d_res = self.conv1d_params2(conv1d_res)
        conv1d_res = F.relu(conv1d_res)

        to_dense = conv1d_res.view(batch_sortpooling_graphs.size(0), -1)

        if self.output_dim > 0:
            out_linear = self.out_params(to_dense)
            reluact_fp = F.relu(out_linear)
        else:
            reluact_fp = to_dense

        return F.relu(reluact_fp)

