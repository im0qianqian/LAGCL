import os
import pandas as pd
from collections import defaultdict
import dgl
import torch


def build_node_table(training_set_u, training_set_i):
    res = []
    embedding_idx = 0
    for node_id in sorted(list(training_set_u.keys())):
        res.append({
            'node_id': node_id,
            'node_feature': embedding_idx,
            'node_degree': len(training_set_u.get(node_id, [])),
            'node_type': 0,
        })
        embedding_idx = embedding_idx + 1
    for node_id in sorted(list(training_set_i.keys())):
        res.append({
            'node_id': node_id,
            'node_feature': embedding_idx,
            'node_degree': len(training_set_i.get(node_id, [])),
            'node_type': 1,
        })
        embedding_idx = embedding_idx + 1
    node_table = pd.DataFrame(res)
    return node_table


def __generate_set(training_data, test_data):
    training_set_u = defaultdict(dict)
    training_set_i = defaultdict(dict)
    test_set = defaultdict(dict)
    for user, item in training_data[['node1_id', 'node2_id']].values:
        training_set_u[user][item] = 1
        training_set_i[item][user] = 1
    for user, item in test_data[['node1_id', 'node2_id']].values:
        if user not in training_set_u.keys():
            continue
        if item not in training_set_i.keys():
            continue
        test_set[user][item] = 1
    return training_set_u, training_set_i, test_set


def build_graph(dataset_path):
    # step 1, load train.txt and test.txt
    train_edges = pd.read_csv(os.path.join(dataset_path, 'train.txt'),
                              names=['node1_id', 'node2_id', 'weight'],
                              sep=' ')
    train_edges['node1_id'] = 'userid_' + train_edges['node1_id'].astype(str)
    train_edges['node2_id'] = 'itemid_' + train_edges['node2_id'].astype(str)
    test_edges = pd.read_csv(os.path.join(dataset_path, 'test.txt'),
                             names=['node1_id', 'node2_id', 'weight'],
                             sep=' ')
    test_edges['node1_id'] = 'userid_' + test_edges['node1_id'].astype(str)
    test_edges['node2_id'] = 'itemid_' + test_edges['node2_id'].astype(str)

    # step 2, generate u2i / i2u set
    training_set_u, training_set_i, test_set = __generate_set(
        train_edges, test_edges)

    # step 3, generate node_table
    node_table = build_node_table(training_set_u, training_set_i)

    # step 4, generate dgl graph
    node_table_dict_by_node_id = node_table.set_index('node_id').to_dict(
        orient='index')
    node_table_dict_by_node_idx = node_table.set_index('node_feature').to_dict(
        orient='index')
    u = train_edges['node1_id'].map(
        lambda x: node_table_dict_by_node_id[x]['node_feature'])
    v = train_edges['node2_id'].map(
        lambda x: node_table_dict_by_node_id[x]['node_feature'])

    graph = dgl.to_bidirected(dgl.graph((u, v)))

    graph.ndata['node_feature'] = torch.tensor(
        [i for i in range(len(node_table))])
    graph.ndata['node_degree'] = torch.tensor([
        node_table_dict_by_node_idx[i]['node_degree']
        for i in range(len(node_table))
    ])
    graph.ndata['node_type'] = torch.tensor([
        node_table_dict_by_node_idx[i]['node_type']
        for i in range(len(node_table))
    ])

    return graph, node_table, train_edges, (
        training_set_u, training_set_i,
        test_set), (node_table_dict_by_node_id, node_table_dict_by_node_idx)


if __name__ == '__main__':
    dataset_path = 'datasets/lastfm'

    graph, node_table, train_edges, (
        training_set_u, training_set_i,
        test_set), (node_table_dict_by_node_id,
                    node_table_dict_by_node_idx) = build_graph(dataset_path)
    print(graph)
    print(node_table)