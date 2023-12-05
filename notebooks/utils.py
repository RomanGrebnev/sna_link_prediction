import math
import torch
import torch_geometric
from torch_geometric.deprecation import deprecated
from torch_geometric.utils import to_undirected

from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit
from torch_geometric.utils import negative_sampling

def train_test_split_edges_c(
    data: 'torch_geometric.data.Data',
    val_ratio: float = 0.05,
    test_ratio: float = 0.1,
) -> 'torch_geometric.data.Data':
    r"""Splits the edges of a :class:`torch_geometric.data.Data` object
    into positive and negative train/val/test edges.
    As such, it will replace the :obj:`edge_index` attribute with
    :obj:`train_pos_edge_index`, :obj:`train_pos_neg_adj_mask`,
    :obj:`val_pos_edge_index`, :obj:`val_neg_edge_index` and
    :obj:`test_pos_edge_index` attributes.
    If :obj:`data` has edge features named :obj:`edge_attr`, then
    :obj:`train_pos_edge_attr`, :obj:`val_pos_edge_attr` and
    :obj:`test_pos_edge_attr` will be added as well.

    .. warning::

        :meth:`~torch_geometric.utils.train_test_split_edges` is deprecated and
        will be removed in a future release.
        Use :class:`torch_geometric.transforms.RandomLinkSplit` instead.

    Args:
        data (Data): The data object.
        val_ratio (float, optional): The ratio of positive validation edges.
            (default: :obj:`0.05`)
        test_ratio (float, optional): The ratio of positive test edges.
            (default: :obj:`0.1`)

    :rtype: :class:`torch_geometric.data.Data`
    """

    assert 'batch' not in data  # No batch-mode.

    num_nodes = data.num_nodes
    row, col = data.edge_index
    edge_attr = data.edge_attr
    data.edge_index = data.edge_attr = None

    # Return upper triangular portion.
    # mask = row < col
    # row, col = row[mask], col[mask]

    # if edge_attr is not None:
    #     edge_attr = edge_attr[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]
    if edge_attr is not None:
        edge_attr = edge_attr[perm]

    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    if edge_attr is not None:
        data.val_pos_edge_attr = edge_attr[:n_v]

    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)
    if edge_attr is not None:
        data.test_pos_edge_attr = edge_attr[n_v:n_v + n_t]

    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    # if edge_attr is not None:
    #     out = to_undirected(data.train_pos_edge_index, edge_attr[n_v + n_t:])
    #     data.train_pos_edge_index, data.train_pos_edge_attr = out
    # else:
    #     data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)

    # Negative edges.
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0

    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    perm = torch.randperm(neg_row.size(0))[:n_v + n_t]
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    neg_adj_mask[neg_row, neg_col] = 0
    data.train_neg_adj_mask = neg_adj_mask

    row, col = neg_row[:n_v], neg_col[:n_v]
    data.val_neg_edge_index = torch.stack([row, col], dim=0)

    row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
    data.test_neg_edge_index = torch.stack([row, col], dim=0)

    return data



from torch_geometric.utils import (negative_sampling, add_self_loops,
                                   train_test_split_edges)

import random
import math
from copy import copy




def do_edge_split(dataset, fast_split=False, val_ratio=0.05, test_ratio=0.1):
    """From https://github.com/facebookresearch/SEAL_OGB/blob/374bd4424968d21f209618602ed9d9338ac607ab/utils.py#L190"""
    data = copy(dataset)
    random.seed(234)
    torch.manual_seed(234)

    if not fast_split:
        data = train_test_split_edges_c(data, val_ratio, test_ratio)
        edge_index, _ = add_self_loops(data.train_pos_edge_index)
        data.train_neg_edge_index = negative_sampling(
            edge_index, num_nodes=data.num_nodes,
            num_neg_samples=data.train_pos_edge_index.size(1))
        print(data)
    
        
    else:
        num_nodes = data.num_nodes
        print("num_nodes: ", num_nodes)
        row, col = data.edge_index
        print("row: ", row)
        print("col: ", col)
        # Return upper triangular portion.
        mask = row < col
        row, col = row[mask], col[mask]
        n_v = int(math.floor(val_ratio * row.size(0)))
        n_t = int(math.floor(test_ratio * row.size(0)))
        # Positive edges.
        perm = torch.randperm(row.size(0))
        row, col = row[perm], col[perm]
        r, c = row[:n_v], col[:n_v]
        # print("r: ", r)
        # print("c: ", c)
        data.val_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
        data.test_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v + n_t:], col[n_v + n_t:]
        data.train_pos_edge_index = torch.stack([r, c], dim=0)
        # Negative edges (cannot guarantee (i,j) and (j,i) won't both appear)
        neg_edge_index = negative_sampling(
            data.edge_index, num_nodes=num_nodes,
            num_neg_samples=row.size(0))
        data.val_neg_edge_index = neg_edge_index[:, :n_v]
        data.test_neg_edge_index = neg_edge_index[:, n_v:n_v + n_t]
        data.train_neg_edge_index = neg_edge_index[:, n_v + n_t:]

    # split_edge = {'train': {}, 'valid': {}, 'test': {}}
    # split_edge['train']['edge'] = data.train_pos_edge_index.t()
    # print( split_edge['train']['edge'].shape)
    # split_edge['train']['edge_neg'] = data.train_neg_edge_index.t()
    # split_edge['valid']['edge'] = data.val_pos_edge_index.t()
    # print( split_edge['valid']['edge'].shape)
    # split_edge['valid']['edge_neg'] = data.val_neg_edge_index.t()
    # split_edge['test']['edge'] = data.test_pos_edge_index.t()
    # print( split_edge['test']['edge'].shape)
    # split_edge['test']['edge_neg'] = data.test_neg_edge_index.t()

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = data.train_pos_edge_index
    split_edge['train']['edge_neg'] = data.train_neg_edge_index
    split_edge['valid']['edge'] = data.val_pos_edge_index
    split_edge['valid']['edge_neg'] = data.val_neg_edge_index
    split_edge['test']['edge'] = data.test_pos_edge_index
    split_edge['test']['edge_neg'] = data.test_neg_edge_index
    print( split_edge['train']['edge'].shape)
    print( split_edge['valid']['edge'].shape)
    print( split_edge['test']['edge'].shape)
    return split_edge





def create_train_test_split_rls(graph):
    tfs = RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True, add_negative_train_samples=True, neg_sampling_ratio=1.0)  
    graph_train, graph_val, graph_test = tfs(graph)
    pos_train_mask = graph_train.edge_label == 1
    neg_train_mask = graph_train.edge_label == 0

    pos_val_mask = graph_val.edge_label == 1
    neg_val_mask = graph_val.edge_label == 0

    pos_test_mask = graph_test.edge_label == 1
    neg_test_mask = graph_test.edge_label == 0

    def apply_mask(graph, mask):
        return torch.tensor(list(zip(graph.edge_label_index[0][mask], graph.edge_label_index[1][mask]))).T

    dataset = {} 
    dataset['train'] = {}
    dataset['train']['edge'] = apply_mask(graph_train, pos_train_mask)
    dataset['valid'] = {}
    dataset['valid']['edge'] = apply_mask(graph_val, pos_val_mask)
    dataset['valid']['edge_neg'] = apply_mask(graph_val, neg_val_mask)
    dataset['test'] = {}
    dataset['test']['edge'] = apply_mask(graph_test, pos_test_mask)
    dataset['test']['edge_neg'] = apply_mask(graph_test,neg_test_mask)

    print("Train edges: ", dataset['train']['edge'].shape)
    print("Valid positive edges: ", dataset['valid']['edge'].shape)
    print("Valid negative edges: ", dataset['valid']['edge_neg'].shape)
    print("Test positive edges: ", dataset['test']['edge'].shape)
    print("Test negative edges: ", dataset['test']['edge_neg'].shape)
    return dataset

