import numpy as np
import networkx as nx
import pandas as pd
import torch

def common_neighbors(G: nx.Graph, pairs):
    return [(pairs[i][0], pairs[i][1],len(list(nx.common_neighbors(G, pairs[i][0], pairs[i][1])))) for i in range(len(pairs))]

def compute_similarity(G: nx.Graph, method: str):
    """_summary_

    Args:
        G (nx.Graph): an undirected graph
        method (str): string defining the similarity method to use. Possible values are: "common_neighbors", "jaccard", "adamic_adar", "preferential_attachment", "resource_allocation"

    Raises:
        ValueError: In case a not supported method is passed

    Returns:
        Similarity matrix (np.array): Contains the similarity between each pair of nodes in the graph
    """
    similarity_matrix = np.zeros((len(G.nodes()), len(G.nodes())), dtype=float)

    match method:
        case "common_neigbhors":
            method_func = common_neighbors
        case "jaccard":
            method_func = nx.jaccard_coefficient
        case "adamic_adar":
            method_func = nx.adamic_adar_index
        case "preferential_attachment":
            method_func = nx.preferential_attachment
        case "resource_allocation":
            method_func = nx.resource_allocation_index
        case _:
            raise ValueError("Unknown method")

    for i, n1 in enumerate(G.nodes()):
        if i % 500 == 0:
            print("i=",i, "len(G.nodes())=", len(G.nodes()))
    
        for j, n2 in enumerate(G.nodes()):
            if n1 != n2:
                similarity_matrix[i][j] = list(method_func(G, [(n1, n2)]))[0][2]
                
    return similarity_matrix

def map_index_to_node(G, index):
    return list(G.nodes())[index]

def map_node_to_index(G, node):
    return list(G.nodes()).index(node)

def evaluate(G : nx.DiGraph, n_to_cut: int, method: str):
    """_summary_

    Args:
        G (nx.DiGraph): A directed graph containing the edges to cut
        n_to_cut (int): A number of edges to cut from the graph
        method (str): A method to use to compute the similarity matrix

    Returns:
        _type_: _description_
    """
    # cut n_to_cut edges from G
    eval_G = G.copy().to_undirected()
    # generate a list of n_to_cut unique number at rando mbetwee 0 and to len(G.edges())
    edge_idx = np.random.choice(len(eval_G.edges()), n_to_cut, replace=False)


    cutted_edges = [list(eval_G.edges())[idx] for idx in edge_idx]

    # cut the edges from the graph
    eval_G.remove_edges_from(cutted_edges)
    
    # compute similarity matrix
    #similarity_matrix = np.zeros((len(eval_G.nodes()), len(eval_G.nodes())))
    similarity_matrix = compute_similarity(eval_G, method)
    
    # get the top 10 edges by index1, index2 and similarity
    flattened_matrix = [(i, j, similarity_matrix[i][j]) for i in range(len(similarity_matrix)) for j in range(len(similarity_matrix[0]))]
    
    # Sort the flattened matrix based on the values in descending order
    sorted_flat_matrix = sorted(flattened_matrix, key=lambda x: x[2], reverse=True)
    
    # print where the cutted edges are in the sorted matrix
    for edge in cutted_edges:
        edge = (map_node_to_index(eval_G, edge[0]), map_node_to_index(eval_G, edge[1]))
        for i, (index1, index2, value) in enumerate(sorted_flat_matrix):
            if (index1, index2) == edge:
                print("edge=", edge, "index=", i, "value = ", sorted_flat_matrix[i])
    
        
    return similarity_matrix

def torch_to_Graph(source, pos_target, neg_target):
    """_summary_

    Args:
        source: A tensor containing the source nodes
        pos_target: A tensor containing the target nodes
        neg_target: A tensor containing the similarity values

    Returns:
        G (nx.DiGraph): A directed graph based on the positive and negative edges. The positive edges have a weight of 1, the negative edges have a weight of 0.
    """
    df_pos = pd.DataFrame()
    df_neg = pd.DataFrame()
    
    if type(source) == torch.Tensor:
        source = source.numpy()
    elif type(source) == list:
            source = np.array(source)
    elif type(source) == np.ndarray:
        pass
    else:
        raise ValueError("source is not a valid type")
    
    if type(pos_target) == torch.Tensor:
        pos_target = pos_target.numpy()
    elif type(pos_target) == list:
            pos_target = np.array(pos_target)
    elif type(pos_target) == np.ndarray:
        pass
    else:
        raise ValueError("pos_target is not a valid type")
    
    df_pos["source"] = source
    df_pos["target"] = pos_target
    df_pos["weight"] = 1
    if neg_target is not None:        
        if type(neg_target) == torch.Tensor:
            neg_target = neg_target.numpy()
        elif type(neg_target) == list:
                neg_target = np.array(neg_target)
        elif type(neg_target) == np.ndarray:
            pass
        else:
            raise ValueError("neg_target is not a valid type")
    
        df_neg["source"] = source
        df_neg["target"] = neg_target
        df_neg["weight"] = 0
    
    df = pd.concat([df_pos, df_neg])
    
    G = nx.from_pandas_edgelist(df, 
                            source='source', 
                            target='target', 
                            edge_attr = 'weight',
                            create_using=nx.DiGraph())
    return G