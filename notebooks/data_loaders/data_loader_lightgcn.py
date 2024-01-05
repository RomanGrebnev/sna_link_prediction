import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from torch import torch


import random

from sklearn.model_selection import train_test_split

import torch
from typing import Dict
from torch_sparse import SparseTensor

from torch_geometric.utils.num_nodes import maybe_num_nodes


class UserUserDatasetLightGCN:
    """Class to create user-user interaction dataset from postings and votes dataframes

    Input:
    datasets_dict: dict of dataframes with postings and votes
    verbose: print statements

    Methods:
    _create_user_user_dataframe: creates user-user interaction dataframe
    _create_train_test_val_split: creates train, test and val splits
    _create_negative_samples: creates negative samples for training
    _create_random_mini_batches: creates random mini batches for training
    get_train_test_val_split: returns train, test and val splits
    get_negative_samples: returns negative samples for training
    get_random_mini_batches: returns random mini batches for training
    """

    def __init__(
        self,
        datasets_dict: Dict[
            str, pd.DataFrame
        ],  # dict of dataframes with postings and votes
        verbose: bool = True,
    ):
        self.dataset_dict = datasets_dict
        self.verbose = verbose
        self._create_user_article_dataframe()  # creates mappings and edge_index

    def _create_user_article_dataframe(self):
        """Creates user-article interaction dataframe from postings and votes dataframes"""
        if self.dataset_dict.get("postings") is not None:
            postings = self.dataset_dict.get("postings")
        else:
            raise ValueError("No postings dataframe in datasets_dict")
        if self.dataset_dict.get("votes") is not None:
            votes = self.dataset_dict.get("votes")
        else:
            raise ValueError("No votes dataframe in datasets_dict")

        post_article = postings.groupby(by=["ID_CommunityIdentity", "ID_Article"]).agg(
            {"ArticlePublishingDate": "min"}
        )
        post_article = post_article.reset_index()
        votes_trunc = votes[["ID_CommunityIdentity", "ID_Posting"]]
        postings_trunc = (
            postings.groupby(by=["ID_Posting", "ID_Article"])
            .agg({"ArticlePublishingDate": "min"})
            .reset_index()
            .drop(columns=["ArticlePublishingDate"])
        )
        vote_user_article = votes_trunc.merge(
            postings_trunc, left_on=["ID_Posting"], right_on=["ID_Posting"]
        )
        vote_article = (
            vote_user_article.groupby(by=["ID_CommunityIdentity", "ID_Article"])
            .agg({"ID_Posting": "count"})
            .reset_index()
            .drop(columns=["ID_Posting"])
        )

        self.user_article_interaction = pd.concat([post_article, vote_article]).drop(
            columns=["ArticlePublishingDate"]
        )

        self.user_article_interaction = (
            self.user_article_interaction.groupby(
                by=["ID_CommunityIdentity", "ID_Article"]
            )
            .count()
            .reset_index()
        )

        self.user_article_interaction["user_id"] = "u-" + self.user_article_interaction[
            "ID_CommunityIdentity"
        ].astype(str)
        self.user_article_interaction[
            "article_id"
        ] = "a-" + self.user_article_interaction["ID_Article"].astype(str)

        self.users_mapping = {
            index: i
            for i, index in enumerate(
                np.unique(self.user_article_interaction["user_id"])
            )
        }
        self.articles_mapping = {
            index: i
            for i, index in enumerate(
                np.unique(self.user_article_interaction["article_id"])
            )
        }

        self.user_article_interaction["user_id_enc"] = self.user_article_interaction[
            "user_id"
        ].map(self.users_mapping)
        self.user_article_interaction["article_enc"] = self.user_article_interaction[
            "article_id"
        ].map(self.articles_mapping)

        self.edge_index = torch.stack(
            [
                torch.tensor(np.array(self.user_article_interaction["user_id_enc"])),
                torch.tensor(np.array(self.user_article_interaction["article_enc"])),
            ]
        )

        self.num_users, self.num_articles = len(
            np.unique(self.user_article_interaction["user_id_enc"])
        ), len(np.unique(self.user_article_interaction["article_enc"]))
        self.num_interactions = self.edge_index.shape[1]

        if self.verbose:
            print(
                "Size of user-article interaction dataframe: ",
                self.user_article_interaction.shape,
            )
            print(
                "Number of unique users: ",
                len(np.unique(self.user_article_interaction["user_id"])),
            )
            print(
                "Number of unique articles: ",
                len(np.unique(self.user_article_interaction["article_id"])),
            )
            print("Number of edges: ", self.edge_index.shape[1])

    def _create_train_test_val_split(
        self, val_split_ratio=0.2, test_split_ratio=0.5, random_state=1, sparse=True
    ):
        """Creates train, test and val splits
        Returns: long tensor of shape (2, num_edges) with train, validation and test splits. Depending on sparse, returns SparseTensor or tensor
        """
        all_indices = [i for i in range(self.num_interactions)]

        # create train
        self.train_indices, self.test_indices = train_test_split(
            all_indices, test_size=val_split_ratio, random_state=random_state
        )
        # create val and test
        self.val_indices, self.test_indices = train_test_split(
            self.test_indices, test_size=test_split_ratio, random_state=random_state
        )

        self.train_edge_index = self.edge_index[:, self.train_indices]
        self.val_edge_index = self.edge_index[:, self.val_indices]
        self.test_edge_index = self.edge_index[:, self.test_indices]

        self.train_edge_index = self.train_edge_index.to(torch.long)
        self.val_edge_index = self.val_edge_index.to(torch.long)
        self.test_edge_index = self.test_edge_index.to(torch.long)

        if self.verbose:
            print("Train edge index shape: ", self.train_edge_index.shape)
            print("Val edge index shape: ", self.val_edge_index.shape)
            print("Test edge index shape: ", self.test_edge_index.shape)

        self.train_sparse_edge_index = SparseTensor(
            row=self.train_edge_index[0],
            col=self.train_edge_index[1],
            sparse_sizes=(
                self.num_users + self.num_articles,
                self.num_users + self.num_articles,
            ),
        )
        self.val_sparse_edge_index = SparseTensor(
            row=self.val_edge_index[0],
            col=self.val_edge_index[1],
            sparse_sizes=(
                self.num_users + self.num_articles,
                self.num_users + self.num_articles,
            ),
        )
        self.test_sparse_edge_index = SparseTensor(
            row=self.test_edge_index[0],
            col=self.test_edge_index[1],
            sparse_sizes=(
                self.num_users + self.num_articles,
                self.num_users + self.num_articles,
            ),
        )

        if sparse:
            return (
                self.train_sparse_edge_index,
                self.val_sparse_edge_index,
                self.test_sparse_edge_index,
            )
        else:
            return self.train_edge_index, self.val_edge_index, self.test_edge_index

    def get_train_test_val_split(
        self, val_split_ratio=0.2, test_split_ratio=0.5, random_state=1, sparse=True
    ):
        return self._create_train_test_val_split(
            val_split_ratio, test_split_ratio, random_state, sparse
        )

    def _create_negative_samples(self, edge_index, contains_neg_self_loops=False):
        """Creates negative samples for training
        Returns: torch.tensor of shape (3, num_edges) with positive and negative samples
        """
        num_nodes = maybe_num_nodes(edge_index)
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

        self.source_pos_ind = edge_index[0]
        self.target_pos_ind = edge_index[1]
        self.target_neg_ind = unique_nodes[rand].to(edge_index.device)

        return torch.stack(
            (self.source_pos_ind, self.target_pos_ind, self.target_neg_ind), dim=0
        )

    def get_negative_samples(self, edge_index, contains_neg_self_loops=False):
        """Returns positive user indices, positive article indices and negative article indices for training"""
        return self._create_negative_samples(edge_index, contains_neg_self_loops)

    def _create_random_mini_batches(self, batch_size):
        """Creates random mini batches for training"""
        sampled_edges = torch.stack(
            (self.source_pos_ind, self.target_pos_ind, self.target_neg_ind), dim=0
        )
        indices = random.choices(
            [i for i in range(sampled_edges[0].shape[0])], k=batch_size
        )
        batch = sampled_edges[:, indices]
        user_indices, pos_item_indices, neg_item_indices = batch[0], batch[1], batch[2]
        return user_indices, pos_item_indices, neg_item_indices

    def get_random_mini_batches(self, batch_size):
        """Returns random mini batches for training"""
        return self._create_random_mini_batches(batch_size)
