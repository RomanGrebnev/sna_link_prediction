import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit
from typing import Optional, Tuple, Union
from torch_geometric.utils.num_nodes import maybe_num_nodes
import matplotlib.pyplot as plt
import numpy as np

import subprocess as sp
import os
from tqdm import tqdm


class GNNStack(torch.nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, num_layers, dropout, emb=False
    ):
        super(GNNStack, self).__init__()

        # GraphSAGE convolutional layers
        conv_model = pyg.nn.SAGEConv

        self.convs = nn.ModuleList()

        # add the first convolutional layer
        self.convs.append(conv_model(input_dim, hidden_dim))
        self.dropout = dropout
        self.num_layers = num_layers
        self.emb = emb

        # Create num_layers GraphSAGE convs
        assert self.num_layers >= 1, "Number of layers is not >=1"
        for l in range(self.num_layers - 1):
            self.convs.append(conv_model(hidden_dim, hidden_dim))

        # post-message-passing processing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.post_mp(x)

        # Return final layer of embeddings if specified
        if self.emb:
            return x

        # Else return class probabilities
        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


class LinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(LinkPredictor, self).__init__()

        # Create linear layers
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        # x_i and x_j are both of shape (E, D)
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def calculate_roc_auc(pos_preds, neg_preds):
    # Stack predictions and labels together
    preds = torch.cat([pos_preds, neg_preds])
    labels = torch.cat([torch.ones(pos_preds.size(0)), torch.zeros(neg_preds.size(0))])

    # Use sklearn's roc auc function
    from sklearn.metrics import roc_auc_score

    return roc_auc_score(labels.detach().cpu(), preds.detach().cpu())


def structured_negative_sampling(
    edge_index, num_nodes: Optional[int] = None, contains_neg_self_loops: bool = True
):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index.cpu()

    pos_idx = row * num_nodes + col
    if not contains_neg_self_loops:
        loop_idx = torch.arange(num_nodes) * (num_nodes + 1)
        pos_idx = torch.cat([pos_idx, loop_idx], dim=0)

    # Get the unique nodes from the columns of edge_index
    unique_nodes = torch.unique(edge_index[1]).cpu()

    # Generate random indices for negative edges from unique_nodes
    rand = torch.randint(len(unique_nodes), (row.size(0),), dtype=torch.long)
    neg_idx = row * num_nodes + unique_nodes[rand]

    mask = torch.from_numpy(np.isin(neg_idx, pos_idx)).to(torch.bool)
    rest = mask.nonzero(as_tuple=False).view(-1)

    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.randint(len(unique_nodes), (rest.size(0),), dtype=torch.long)
        rand[rest] = tmp
        neg_idx = row[rest] * num_nodes + unique_nodes[tmp]

        mask = torch.from_numpy(np.isin(neg_idx, pos_idx)).to(torch.bool)
        rest = rest[mask]

    return torch.stack(
        [edge_index[0], torch.tensor(unique_nodes[rand]).to(edge_index.device)], dim=0
    ).long()


def train(
    model,
    link_predictor,
    emb,
    edge_index,
    pos_train_edge,
    batch_size,
    optimizer,
    debug=False,
):
    """
    Runs offline training for model, link_predictor and node embeddings given the message
    edges and supervision edges.
    :param model: Torch Graph model used for updating node embeddings based on message passing
    :param link_predictor: Torch model used for predicting whether edge exists or not
    :param emb: (N, d) Initial node embeddings for all N nodes in graph
    :param edge_index: (2, E) Edge index for all edges in the graph
    :param pos_train_edge: (PE, 2) Positive edges used for training supervision loss
    :param batch_size: Number of positive (and negative) supervision edges to sample per batch
    :param optimizer: Torch Optimizer to update model parameters
    :return: Average supervision loss over all positive (and correspondingly sampled negative) edges
    """
    model.train()
    link_predictor.train()
    train_losses = []

    pos_train_edge = torch.tensor(pos_train_edge, dtype=torch.long).T
    neg_train_edge = structured_negative_sampling(
        pos_train_edge.T, num_nodes=pos_train_edge.shape[0]
    ).T
    dataloader = DataLoader(range(pos_train_edge.shape[0]), batch_size, shuffle=True)

    for edge_id in tqdm(dataloader):
        optimizer.zero_grad()

        # Run message passing on the inital node embeddings to get updated embeddings
        node_emb = model(emb, edge_index)  # (N, d)

        # Predict the class probabilities on the batch of positive edges using link_predictor
        pos_edge = pos_train_edge[edge_id].T  # (2, B)
        pos_pred = link_predictor(node_emb[pos_edge[0]], node_emb[pos_edge[1]])  # (B, )

        neg_edge = neg_train_edge[edge_id].T  # (2, B)
        neg_pred = link_predictor(node_emb[neg_edge[0]], node_emb[neg_edge[1]])  # (Ne,)

        # Compute the corresponding negative log likelihood loss on the positive and negative edges
        loss = (
            -torch.log(pos_pred + 1e-15).mean() - torch.log(1 - neg_pred + 1e-15).mean()
        )

        # Backpropagate and update parameters
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        if debug:
            print("Train edges batch shape: ", edge_id.shape)
            print("Train node embedding shape: ", node_emb.shape)
            print("Train positive edge shape: ", pos_edge.shape)
            print("Shape of positive edges (source): ", pos_edge[0].shape)
            print("Shape of positive edges (target): ", pos_edge[1].shape)
            print("Node embedding shape: ", node_emb[pos_edge[0]].shape)
            print("Train positive prediction shape: ", pos_pred.shape)
            print("Train negative prediction shape: ", neg_pred.shape)
            print("------------------------------------------")

    print("Train loss: ", sum(train_losses) / len(train_losses))

    return sum(train_losses) / len(train_losses)


def test(model, predictor, emb, edge_index, split_edge, batch_size, evaluator):
    """
    Evaluates graph model on validation and test edges
    :param model: Torch Graph model used for updating node embeddings based on message passing
    :param predictor: Torch model used for predicting whether edge exists or not
    :param emb: (N, d) Initial node embeddings for all N nodes in graph
    :param edge_index: (2, E) Edge index for all edges in the graph
    :param split_edge: Dictionary of (e, 2) edges for val pos/neg and test pos/neg edges
    :param batch_size: Number of positive (and negative) supervision edges to sample per batch
    :param evaluator: OGB evaluator to calculate hits @ k metric
    :return: hits @ k results
    """
    model.eval()
    predictor.eval()

    node_emb = model(emb, edge_index)

    pos_valid_edge = split_edge["valid"]["edge"].T.to(emb.device)
    neg_valid_edge = split_edge["valid"]["edge_neg"].T.to(emb.device)
    pos_test_edge = split_edge["test"]["edge"].T.to(emb.device)
    neg_test_edge = split_edge["test"]["edge_neg"].T.to(emb.device)

    with torch.no_grad():
        loss_val_pos = []
        pos_valid_preds = []
        for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
            edge = pos_valid_edge[perm].t()
            pos_valid_preds += [
                predictor(node_emb[edge[0]], node_emb[edge[1]]).squeeze().cpu()
            ]
            loss_val_pos.append(
                -torch.log(
                    predictor(node_emb[edge[0]], node_emb[edge[1]]).squeeze().cpu()
                    + 1e-15
                ).mean()
            )

        pos_valid_pred = torch.cat(pos_valid_preds, dim=0)
        loss_val_pos_val = sum(loss_val_pos) / len(loss_val_pos)

        loss_val_neg = []
        neg_valid_preds = []
        for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
            edge = neg_valid_edge[perm].t()
            neg_valid_preds += [
                predictor(node_emb[edge[0]], node_emb[edge[1]]).squeeze().cpu()
            ]
            loss_val_neg.append(
                -torch.log(
                    1
                    - predictor(node_emb[edge[0]], node_emb[edge[1]]).squeeze().cpu()
                    + 1e-15
                ).mean()
            )

        neg_valid_pred = torch.cat(neg_valid_preds, dim=0)
        loss_val_neg_val = sum(loss_val_neg) / len(loss_val_neg)

        loss_test_pos = []
        pos_test_preds = []
        for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
            edge = pos_test_edge[perm].t()
            pos_test_preds += [
                predictor(node_emb[edge[0]], node_emb[edge[1]]).squeeze().cpu()
            ]
            loss_test_pos.append(
                -torch.log(
                    predictor(node_emb[edge[0]], node_emb[edge[1]]).squeeze().cpu()
                    + 1e-15
                ).mean()
            )

        pos_test_pred = torch.cat(pos_test_preds, dim=0)
        loss_test_pos_val = sum(loss_test_pos) / len(loss_test_pos)

        loss_test_neg = []
        neg_test_preds = []
        for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
            edge = neg_test_edge[perm].t()
            neg_test_preds += [
                predictor(node_emb[edge[0]], node_emb[edge[1]]).squeeze().cpu()
            ]
            loss_test_neg.append(
                -torch.log(
                    1
                    - predictor(node_emb[edge[0]], node_emb[edge[1]]).squeeze().cpu()
                    + 1e-15
                ).mean()
            )

        neg_test_pred = torch.cat(neg_test_preds, dim=0)
        loss_test_neg_val = sum(loss_test_neg) / len(loss_test_neg)

        results = {}
        results["loss_val_pos"] = loss_val_pos_val
        results["loss_val_neg"] = loss_val_neg_val
        results["loss_test_pos"] = loss_test_pos_val
        results["loss_test_neg"] = loss_test_neg_val
        results["roc_auc_val"] = calculate_roc_auc(pos_valid_pred, neg_valid_pred)
        results["roc_auc_test"] = calculate_roc_auc(pos_test_pred, neg_test_pred)

        print("Loss val pos: ", loss_val_pos_val)
        print("Loss val neg: ", loss_val_neg_val)
        print("Loss test pos: ", loss_test_pos_val)
        print("Loss test neg: ", loss_test_neg_val)
        print("ROC-AUC val: ", calculate_roc_auc(pos_valid_pred, neg_valid_pred))
        print("ROC-AUC test: ", calculate_roc_auc(pos_test_pred, neg_test_pred))

        for K in [20, 50, 100]:
            evaluator.K = K
            valid_hits = evaluator.eval(
                {
                    "y_pred_pos": pos_valid_pred,
                    "y_pred_neg": neg_valid_pred,
                }
            )[f"hits@{K}"]
            test_hits = evaluator.eval(
                {
                    "y_pred_pos": pos_test_pred,
                    "y_pred_neg": neg_test_pred,
                }
            )[f"hits@{K}"]

            results[f"Hits@{K}"] = (valid_hits, test_hits)

    return results
