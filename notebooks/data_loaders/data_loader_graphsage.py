from typing import Dict
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from torch import torch
from sklearn import preprocessing

from torch_geometric.data import Data
from torch_geometric.transforms.to_undirected import ToUndirected

from sklearn.model_selection import train_test_split

from typing import Optional, Tuple, Union
from torch_geometric.utils.num_nodes import maybe_num_nodes


class UserUserDatasetGraphSAGE:
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

    def _create_user_article_dataframe(self, subsampling_ratio=0.5):
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

        if self.verbose:
            print(
                f"User-article interaction dataframe created with {self.user_article_interaction.shape[0]} rows"
            )

        # self.user_article_interaction = self.user_article_interaction.iloc[
        #     : int(self.user_article_interaction.shape[0] * subsampling_ratio), :
        # ]

        self.user_article_interaction = self.user_article_interaction.sample(
            n=int(self.user_article_interaction.shape[0] * subsampling_ratio),
            random_state=1,
        )

        if self.verbose:
            print(
                f"User-article interaction dataframe subsampled to {self.user_article_interaction.shape[0]} rows"
            )

        # concatenate nodes and create node encodings
        self.nodes = np.concatenate(
            (
                np.unique(self.user_article_interaction["user_id"]),
                np.unique(self.user_article_interaction["article_id"]),
            )
        )
        self.nodes_user = np.unique(self.user_article_interaction["user_id"])
        self.nodes_article = np.unique(self.user_article_interaction["article_id"])

        self.label_encoder_nodes = preprocessing.LabelEncoder()
        self.label_encoder_nodes.fit(self.nodes)
        self.nodes_enc = self.label_encoder_nodes.transform(self.nodes)

        self.user_article_interaction[
            "user_id_enc"
        ] = self.label_encoder_nodes.transform(self.user_article_interaction["user_id"])
        self.user_article_interaction[
            "article_enc"
        ] = self.label_encoder_nodes.transform(
            self.user_article_interaction["article_id"]
        )

        # generate edge list
        edge_list = []
        for row in self.user_article_interaction.itertuples():
            edge_list.append([row.user_id_enc, row.article_enc])

        # create tensor with nodes
        self.nodes_graph = torch.tensor(self.nodes_enc, dtype=torch.long)

        # create tensor with edges
        self.edge_index = torch.tensor(edge_list, dtype=torch.long).t()

        # initialize graph with nodes and edges
        x = torch.tensor(self.nodes_enc, dtype=torch.long)
        self.graph = Data(x=x, edge_index=self.edge_index)
        # self.graph = ToUndirected()(self.graph)

        # validate graph
        self.graph.validate(raise_on_error=True)

        if self.verbose:
            print(
                f"Graph created with {self.graph.num_nodes} nodes and {self.graph.num_edges} edges"
            )

    @staticmethod
    def structured_negative_sampling(
        edge_index,
        num_nodes: Optional[int] = None,
        contains_neg_self_loops: bool = True,
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
            [edge_index[0], torch.tensor(unique_nodes[rand]).to(edge_index.device)],
            dim=0,
        ).long()

    def create_train_test_split(self, num_val=0.2, num_test=0.5):
        num_users, num_articles = len(np.unique(self.edge_index[0])), len(
            np.unique(self.edge_index[1])
        )
        print(num_users, num_articles)
        num_interactions = self.edge_index.shape[1]
        neg_samples = self.structured_negative_sampling(
            self.graph.edge_index, num_nodes=self.edge_index.shape[1]
        )

        all_indices = [i for i in range(num_interactions)]

        train_indices, test_indices = train_test_split(
            all_indices, test_size=0.2, random_state=1
        )
        val_indices, test_indices = train_test_split(
            test_indices, test_size=0.5, random_state=1
        )

        train_edge_index = self.edge_index[:, train_indices]
        val_edge_index = self.edge_index[:, val_indices]
        test_edge_index = self.edge_index[:, test_indices]

        train_edge_index = train_edge_index.to(torch.long)
        val_edge_index = val_edge_index.to(torch.long)
        test_edge_index = test_edge_index.to(torch.long)

        val_neg_edge_index = neg_samples[:, val_indices]
        test_neg_edge_index = neg_samples[:, test_indices]

        dataset = {}
        dataset["train"] = {}
        dataset["train"]["edge"] = train_edge_index
        dataset["valid"] = {}
        dataset["valid"]["edge"] = val_edge_index
        dataset["valid"]["edge_neg"] = val_neg_edge_index
        dataset["test"] = {}
        dataset["test"]["edge"] = test_edge_index
        dataset["test"]["edge_neg"] = test_neg_edge_index

        print("Train edges: ", dataset["train"]["edge"].shape)
        print("Valid positive edges: ", dataset["valid"]["edge"].shape)
        print("Valid negative edges: ", dataset["valid"]["edge_neg"].shape)
        print("Test positive edges: ", dataset["test"]["edge"].shape)
        print("Test negative edges: ", dataset["test"]["edge_neg"].shape)
        return dataset
