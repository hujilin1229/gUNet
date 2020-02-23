from collections import namedtuple

import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_sparse import coalesce
from torch_geometric.nn import GraphConv
from torch_geometric.nn.pool.topk_pool import filter_adj
import torch_geometric as tg
import numpy as np
from torch_geometric.utils import degree

class StarPooling(torch.nn.Module):
    r"""The edge pooling operator from the `"Towards Graph Pooling by Edge
    Contraction" <https://graphreason.github.io/papers/17.pdf>`_ and
    `"Edge Contraction Pooling for Graph Neural Networks"
    <https://arxiv.org/abs/1905.10990>`_ papers.

    In short, a score is computed for each edge.
    Edges are contracted iteratively according to that score unless one of
    their nodes has already been part of a contracted edge.

    To duplicate the configuration from the "Towards Graph Pooling by Edge
    Contraction" paper, use either
    :func:`EdgePooling.compute_edge_score_softmax`
    or :func:`EdgePooling.compute_edge_score_tanh`, and set
    :obj:`add_to_edge_score` to :obj:`0`.

    To duplicate the configuration from the "Edge Contraction Pooling for
    Graph Neural Networks" paper, set :obj:`dropout` to :obj:`0.2`.

    Args:
        in_channels (int): Size of each input sample.
        edge_score_method (function, optional): The function to apply
            to compute the edge score from raw edge scores. By default,
            this is the softmax over all incoming edges for each node.
            This function takes in a :obj:`raw_edge_score` tensor of shape
            :obj:`[num_nodes]`, an :obj:`edge_index` tensor and the number of
            nodes :obj:`num_nodes`, and produces a new tensor of the same size
            as :obj:`raw_edge_score` describing normalized edge scores.
            Included functions are
            :func:`EdgePooling.compute_edge_score_softmax`,
            :func:`EdgePooling.compute_edge_score_tanh`, and
            :func:`EdgePooling.compute_edge_score_sigmoid`.
            (default: :func:`EdgePooling.compute_edge_score_softmax`)
        dropout (float, optional): The probability with
            which to drop edge scores during training. (default: :obj:`0`)
        add_to_edge_score (float, optional): This is added to each
            computed edge score. Adding this greatly helps with unpool
            stability. (default: :obj:`0.5`)
    """

    unpool_description = namedtuple(
        "UnpoolDescription",
        ["edge_index", "cluster", "batch"])

    def __init__(self, in_channels, node_score_method=None, dropout=0):
        super(StarPooling, self).__init__()
        self.in_channels = in_channels
        if node_score_method is None:
            node_score_method = self.compute_node_score_tanh

        self.compute_node_score = node_score_method
        self.dropout = dropout

        self.score_func = GraphConv(in_channels, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.score_func.reset_parameters()


    # @staticmethod
    # def compute_edge_score_softmax(raw_edge_score, edge_index, num_nodes):
    #     return softmax(raw_edge_score, edge_index[1], num_nodes)

    @staticmethod
    def compute_node_score_tanh(raw_edge_score):
        return torch.tanh(raw_edge_score)

    @staticmethod
    def compute_node_score_sigmoid(raw_edge_score):
        return torch.sigmoid(raw_edge_score)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        r"""Forward computation which computes the raw edge score, normalizes
        it, and merges the edges.

        Args:
            x (Tensor): The node features.
            edge_index (LongTensor): The edge indices.
            batch (LongTensor): Batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific example.

        Return types:
            * **x** *(Tensor)* - The pooled node features.
            * **edge_index** *(LongTensor)* - The coarsened edge indices.
            * **batch** *(LongTensor)* - The coarsened batch vector.
            * **unpool_info** *(unpool_description)* - Information that is
              consumed by :func:`EdgePooling.unpool` for unpooling.
        """
        # e = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        # e = self.lin(e).view(-1)
        # e = F.dropout(e, p=self.dropout, training=self.training)
        # e = self.compute_edge_score(e)
        # e = e + self.add_to_edge_score

        # TODO: change linear to consider both node and edge features
        # n = self.lin(x).view(-1)

        n = self.score_func(x, edge_index).view(-1)
        n = F.dropout(n, p=self.dropout, training=self.training)
        n = self.compute_node_score(n)

        x, edge_index, edge_attr, batch, unpool_info, perm = self.__merge_stars_with_attr_gpu2__(
            x, edge_index, batch, edge_attr, n)
        # print(perm)
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=n.size(0))

        return x, edge_index, edge_attr, batch, perm

    def __merge_stars_with_attr_gpu2__(self, x, edge_index, batch, edge_attr, node_score):

        node_argsort = torch.argsort(node_score, descending=True)
        cluster = torch.empty_like(batch, device=torch.device('cpu'))
        nodes_remain = torch.ones_like(batch, device=torch.device('cpu'), dtype=torch.bool)
        # Iterate through all edges, selecting it if it is not incident to another already chosen edge.
        edge_index_cpu = edge_index.cpu()
        i = 0

        print("edge index 0", edge_index_cpu[0])

        degrees = degree(edge_index_cpu[0]).long()
        cum_num_nodes = torch.cat(
            [degrees.new_zeros(1),
             degrees.cumsum(dim=0)[:-1]], dim=0).long()

        center_nodes = set()

        print(degrees)
        print(edge_index_cpu.size())
        print(cum_num_nodes)

        for node_idx in node_argsort.tolist():
            if not nodes_remain[node_idx]:
                continue

            dests = edge_index_cpu[1][cum_num_nodes[node_idx].item():cum_num_nodes[node_idx].item()+degrees[node_idx].item()]

            nodes_remain[dests] = False
            nodes_remain[node_idx] = False

            # add node_idx to center_nodes
            center_nodes.add(node_idx)

            cluster[node_idx] = i
            cluster[dests] = i
            i += 1

        cluster = cluster.to(x.device)
        new_x = scatter_add(x, cluster, dim=0, dim_size=i)
        N = new_x.size(0)

        print(cluster)
        new_edge_index, new_edge_attr = coalesce(cluster[edge_index], edge_attr, N, N)

        new_batch = x.new_empty(new_x.size(0), dtype=torch.long)
        new_batch = new_batch.scatter_(0, cluster, batch)

        unpool_info = self.unpool_description(edge_index=edge_index,
                                              cluster=cluster, batch=batch)

        perm = sorted(center_nodes)
        perm = torch.from_numpy(np.array(perm)).view(-1).to(x.device)

        return new_x, new_edge_index, new_edge_attr, new_batch, unpool_info, perm

    def __merge_stars_with_attr_gpu__(self, x, edge_index, edge_attr, batch, node_score):

        device = x.device

        nodes_remaining = set(range(x.size(0)))
        node_argsort = torch.argsort(node_score, descending=True)

        cluster = torch.empty_like(batch, device=torch.device('cpu'))

        # Iterate through all edges, selecting it if it is not incident to another already chosen edge.
        edge_index_cpu = edge_index.cpu()
        center_nodes = set()
        i = 0

        for node_idx in node_argsort.tolist():
            if node_idx not in nodes_remaining:
                continue
            dest_bool = edge_index_cpu[0] == node_idx
            # get the connected nodes
            dests = set(edge_index_cpu[1][dest_bool].numpy())
            # remove the previous combined nodes
            dests.difference_update(center_nodes)
            nodes_remaining.difference_update(dests)
            nodes_remaining.remove(node_idx)

            # add node_idx to center_nodes
            center_nodes.add(node_idx)

            cluster[node_idx] = i
            cluster[list(dests)] = i
            i += 1

        # The remaining nodes are simply kept.
        for node_idx in nodes_remaining:
            cluster[node_idx] = i
            i += 1

        cluster = cluster.to(x.device)

        new_x = scatter_add(x, cluster, dim=0, dim_size=i)
        N = new_x.size(0)

        new_edge_index, new_edge_attr = coalesce(cluster[edge_index], edge_attr, N, N)

        new_batch = x.new_empty(new_x.size(0), dtype=torch.long)
        new_batch = new_batch.scatter_(0, cluster, batch)

        unpool_info = self.unpool_description(edge_index=edge_index,
                                              cluster=cluster, batch=batch)

        perm = sorted(center_nodes)
        perm = torch.from_numpy(np.array(perm)).view(-1).to(device)

        return new_x, new_edge_index, new_edge_attr, new_batch, unpool_info, perm

    def unpool(self, x, unpool_info):
        r"""Unpools a previous edge pooling step.

        For unpooling, :obj:`x` should be of same shape as those produced by
        this layer's :func:`forward` function. Then, it will produce an
        unpooled :obj:`x` in addition to :obj:`edge_index` and :obj:`batch`.

        Args:
            x (Tensor): The node features.
            unpool_info (unpool_description): Information that has
                been produced by :func:`EdgePooling.forward`.

        Return types:
            * **x** *(Tensor)* - The unpooled node features.
            * **edge_index** *(LongTensor)* - The new edge indices.
            * **batch** *(LongTensor)* - The new batch vector.
        """

        new_x = x / unpool_info.new_edge_score.view(-1, 1)
        new_x = new_x[unpool_info.cluster]
        return new_x, unpool_info.edge_index, unpool_info.batch


    def __merge_star_nodes__(self, x, edge_index, batch, node_score):
        """
        Copy from Edge Contraction Pooling

        :param x: node feature
        :param edge_index: edge index
        :param batch: batch index
        :param edge_score: edge score tensor
        :return:
        """

        nodes_remaining = set(range(x.size(0)))

        cluster = torch.empty_like(batch, device=torch.device('cpu'))
        # edge_argsort = torch.argsort(edge_score, descending=True)
        node_argsort = torch.argsort(node_score, descending=True)
        edge_index_cpu = edge_index.cpu()

        deg = tg.utils.degree(edge_index_cpu[0], x.size(0))



        # Iterate through all edges, selecting it if it is not incident to
        # another already chosen edge.
        i = 0
        new_edge_indices = []
        edge_index_cpu = edge_index.cpu()

        for node_idx in node_argsort.tolist():
            # check if the node is still in the nodes_remaining
            if node_idx not in nodes_remaining:
                continue

            dest_bool = edge_index_cpu[0] == node_idx
            dests = set(edge_index_cpu[1][dest_bool].numpy())
            dests.difference_update(nodes_remaining)

            if len(dests) == 0:
                continue

            cluster[list(dests)] = i
            nodes_remaining.remove(node_idx)
            nodes_remaining.difference_update(dests)

            i += 1

        cluster = cluster.to(x.device)

        # We compute the new features as an addition of the old ones.
        new_x = scatter_add(x, cluster, dim=0, dim_size=i)
        new_edge_score = edge_score[new_edge_indices]
        if len(nodes_remaining) > 0:
            remaining_score = x.new_ones(
                (new_x.size(0) - len(new_edge_indices), ))
            new_edge_score = torch.cat([new_edge_score, remaining_score])
        new_x = new_x * new_edge_score.view(-1, 1)

        N = new_x.size(0)
        new_edge_index, _ = coalesce(cluster[edge_index], None, N, N)

        new_batch = x.new_empty(new_x.size(0), dtype=torch.long)
        new_batch = new_batch.scatter_(0, cluster, batch)

        unpool_info = self.unpool_description(edge_index=edge_index,
                                              cluster=cluster, batch=batch,
                                              new_edge_score=new_edge_score)

        return new_x, new_edge_index, new_batch, unpool_info


    def __merge_edges__(self, x, edge_index, batch, edge_score):
        """
        Copy from Edge Contraction Pooling

        :param x: node feature
        :param edge_index: edge index
        :param batch: batch index
        :param edge_score: edge score tensor
        :return:
        """
        nodes_remaining = set(range(x.size(0)))

        cluster = torch.empty_like(batch, device=torch.device('cpu'))
        edge_argsort = torch.argsort(edge_score, descending=True)

        # Iterate through all edges, selecting it if it is not incident to
        # another already chosen edge.
        i = 0
        new_edge_indices = []
        edge_index_cpu = edge_index.cpu()
        for edge_idx in edge_argsort.tolist():
            source = edge_index_cpu[0, edge_idx].item()
            if source not in nodes_remaining:
                continue

            target = edge_index_cpu[1, edge_idx].item()
            if target not in nodes_remaining:
                continue

            new_edge_indices.append(edge_idx)

            cluster[source] = i
            nodes_remaining.remove(source)

            if source != target:
                cluster[target] = i
                nodes_remaining.remove(target)

            i += 1

        # The remaining nodes are simply kept.
        for node_idx in nodes_remaining:
            cluster[node_idx] = i
            i += 1
        cluster = cluster.to(x.device)

        # We compute the new features as an addition of the old ones.
        new_x = scatter_add(x, cluster, dim=0, dim_size=i)
        new_edge_score = edge_score[new_edge_indices]
        if len(nodes_remaining) > 0:
            remaining_score = x.new_ones(
                (new_x.size(0) - len(new_edge_indices), ))
            new_edge_score = torch.cat([new_edge_score, remaining_score])
        new_x = new_x * new_edge_score.view(-1, 1)

        N = new_x.size(0)
        new_edge_index, _ = coalesce(cluster[edge_index], None, N, N)

        new_batch = x.new_empty(new_x.size(0), dtype=torch.long)
        new_batch = new_batch.scatter_(0, cluster, batch)

        unpool_info = self.unpool_description(edge_index=edge_index,
                                              cluster=cluster, batch=batch,
                                              new_edge_score=new_edge_score)

        return new_x, new_edge_index, new_batch, unpool_info

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.in_channels)
