import json
from collections import defaultdict

import numpy as np
import pandas as pd
# from ogb.nodeproppred import NodePropPredDataset


class DataCenter(object):
    """docstring for DataCenter"""

    def __init__(self, config):
        super(DataCenter, self).__init__()
        self.config = config

    def get_train_features(self, dataset, path):
        if dataset == "reddit":
            reddit_feats = np.load(path)
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(reddit_feats)
            reddit_feats = scaler.transform(reddit_feats)
            return reddit_feats
        # elif dataset == "arxiv":
        #     data = NodePropPredDataset(name="ogbn-arxiv")
        #     graph, _ = data[0]  # graph: library-agnostic graph object
        #     return graph["node_feat"]
        elif dataset == "papers100m":
            node_feat = pd.read_csv(path, header=None).values
            if 'int' in str(node_feat.dtype):
                node_feat = node_feat.astype(np.int64)
            else:
                # float
                node_feat = node_feat.astype(np.float32)
            return node_feat
        elif dataset == "bitcoin":
            features = pd.read_csv(path, header=None)
            return features.values.astype(np.float32)
        elif dataset == "twitch":
            feature_dim = 3170  # maximum feature id of all twitch datasets
            features_raw = json.load(open(path))
            node2idx = {}
            features = []
            for i, it in enumerate(features_raw.items()):
                node2idx[int(it[0])] = i
                feats = np.zeros(feature_dim)
                feats[[int(val) for val in it[1]]] = 1
                features.append(feats)
            features = np.asarray([np.asarray(f) for f in features])
            return features
        else:
            raise Exception("Unknown dataset")

    def get_train_adj_list(self, dataset):
        if dataset == "reddit":
            reddit_edges = pd.read_csv(self.config['file_path.reddit_edges'])
            adj_train = defaultdict(set)

            for i, r in reddit_edges.iterrows():
                e = r["edge"]
                src = int(e.split(",")[0][1:])
                dst = int(e.split(",")[1][1:-1])
                adj_train[src].add(dst)

            return adj_train
        # elif dataset == "arxiv":
        #     data = NodePropPredDataset(name="ogbn-arxiv")
        #
        #     split_idx = data.get_idx_split()
        #     train_idx = set(split_idx["train"])
        #     graph, label = data[0]  # graph: library-agnostic graph object
        #     adj_list_train = defaultdict(set)
        #     # we limit the training dataset to first 10k edges
        #     for src, dst in zip(graph['edge_index'][0][:10000], graph['edge_index'][1][:10000]):
        #         if src in train_idx and dst in train_idx:
        #             adj_list_train[src].add(dst)
        #     return adj_list_train
        elif dataset == "papers100m":
            edge_file = self.config['file_path.papers100m_edges']
            nodes_file = self.config['file_path.papers100m_nodes']

            edges_raw = pd.read_csv(edge_file)
            edges = edges_raw.values.tolist()
            print(edges[0])
            edges = list(map(lambda x: [int(x[0]), int(x[1])], edges))
            adj_list = defaultdict(set)
            node2idx = {}

            nodes = pd.read_csv(nodes_file).values.tolist()
            nodes = list(map(lambda x: int(x[0]), nodes))
            for i, v in enumerate(nodes):
                node2idx[v] = i

            for e in edges:
                adj_list[node2idx[e[0]]].add(node2idx[e[1]])
            return adj_list

        elif dataset == "bitcoin":
            edge_file = self.config['file_path.bitcoin_edges']
            features = pd.read_csv(self.config['file_path.bitcoin_feats'], header=None)

            edges_raw = pd.read_csv(edge_file, sep=" ")
            edges = edges_raw.values.tolist()
            edges = list(map(lambda x: [int(x[0]), int(x[1])], edges))
            adj_list = defaultdict(set)

            node2idx = {}
            for i, f in features.iterrows():
                node2idx[int(f[0])] = i

            for e in edges:
                adj_list[node2idx[e[0]]].add(node2idx[e[1]])
            return adj_list
        elif dataset == "twitch":
            twitch_feature_file = self.config['file_path.twitch_feats']
            twitch_edge_file = self.config['file_path.twitch_edges']
            adj_list, _, _, _ = self.get_twitch_data(twitch_edge_file, twitch_feature_file)
            return adj_list
        else:
            raise Exception("Unknown dataset")

    def load_dataSet(self, dataset='cora'):
        if dataset == "reddit":
            adj_list_train, train_indexs, feat_data_train, train_edges, edge_labels = self.load_reddit_data()

            setattr(self, dataset + '_train', train_indexs)
            setattr(self, dataset + '_feats_train', feat_data_train)
            setattr(self, dataset + '_adj_list_train', adj_list_train)
            setattr(self, dataset + '_train_edges', train_edges)
            setattr(self, dataset + '_edge_labels', edge_labels)

        elif dataset == "twitch":
            twitch_feature_file = self.config['file_path.twitch_feats']
            twitch_edge_file = self.config['file_path.twitch_edges']
            twitch_edge_labels_file = self.config['file_path.twitch_edge_labels']

            adj_list_train, train_features, train_edges, train_labels = self.get_twitch_data(twitch_edge_file,
                                                                                             twitch_feature_file,
                                                                                             twitch_edge_labels_file)

            setattr(self, dataset + '_train', [int(n) for n in adj_list_train.keys()])
            setattr(self, dataset + '_feats_train', train_features)
            setattr(self, dataset + '_adj_list_train', adj_list_train)
            setattr(self, dataset + '_train_edges', train_edges)
            setattr(self, dataset + '_edge_labels', train_labels)

        elif dataset == 'deezer':
            deezer_feature_file = self.config['file_path.deezer_feats']
            deezer_edge_file = self.config['file_path.deezer_edges']
            deezer_edge_labels_file = self.config['file_path.deezer_edge_labels']

            adj_list_train, train_features, train_edges, train_labels = self.get_deezer_data(deezer_edge_file,
                                                                                             deezer_feature_file,
                                                                                             deezer_edge_labels_file)

            setattr(self, dataset + '_train', [int(n) for n in adj_list_train.keys()])
            setattr(self, dataset + '_feats_train', train_features)
            setattr(self, dataset + '_adj_list_train', adj_list_train)
            setattr(self, dataset + '_train_edges', train_edges)
            setattr(self, dataset + '_edge_labels', train_labels)

        # elif dataset == 'arxiv':
        #     data = NodePropPredDataset(name="ogbn-arxiv")
        #
        #     split_idx = data.get_idx_split()
        #     train_idx = set(split_idx["train"])
        #     graph, label = data[0]  # graph: library-agnostic graph object
        #     # print(graph)
        #     adj_list_train = defaultdict(set)
        #     test_edges = []
        #     # we limit the training dataset to first 10k edges
        #     for src, dst in zip(graph['edge_index'][0][:10000], graph['edge_index'][1][:10000]):
        #         if src in train_idx and dst in train_idx:
        #             adj_list_train[src].add(dst)
        #
        #     for src, dst in zip(graph['edge_index'][0], graph['edge_index'][1]):
        #         test_edges.append([src, dst])
        #
        #     print(adj_list_train)
        #     features = graph["node_feat"]
        #
        #     setattr(self, dataset + '_train', [int(n) for n in adj_list_train.keys()])
        #     setattr(self, dataset + '_feats_train', features)
        #     setattr(self, dataset + '_adj_list_train', adj_list_train)

        elif dataset == 'papers100m':
            edge_file = self.config['file_path.papers100m_edges']
            features = self.get_train_features("papers100m")
            nodes_file = self.config['file_path.papers100m_nodes']
            node2idx = {}

            edges_raw = pd.read_csv(edge_file)
            edges = edges_raw.values.tolist()
            print(edges[0])
            edges = list(map(lambda x: [int(x[0]), int(x[1])], edges))
            adj_list = defaultdict(set)

            nodes = pd.read_csv(nodes_file).values.tolist()
            nodes = list(map(lambda x: int(x[0]), nodes))
            for i, v in enumerate(nodes):
                node2idx[v] = i

            for e in edges:
                adj_list[node2idx[e[0]]].add(node2idx[e[1]])

            setattr(self, dataset + '_train', [int(n) for n in adj_list.keys()])
            setattr(self, dataset + '_feats_train', features)
            setattr(self, dataset + '_adj_list_train', adj_list)

        elif dataset == "bitcoin":
            edge_file = self.config['file_path.bitcoin_edges']
            features = pd.read_csv(self.config['file_path.bitcoin_feats'], header=None)

            edges_raw = pd.read_csv(edge_file, sep=" ")
            edges = edges_raw.values.tolist()
            edges = list(map(lambda x: [int(x[0]), int(x[1])], edges))
            adj_list = defaultdict(set)

            node2idx = {}
            for i, f in features.iterrows():
                node2idx[int(f[0])] = i

            # remove nodes ids and time steps
            features.drop(features.columns[[0, 1]], axis=1, inplace=True)
            features = features.values.astype(np.float32)
            for e in edges:
                adj_list[node2idx[e[0]]].add(node2idx[e[1]])

            setattr(self, dataset + '_train', [int(n) for n in adj_list.keys()])
            setattr(self, dataset + '_feats_train', features)
            setattr(self, dataset + '_adj_list_train', adj_list)

    def get_deezer_data(self, deezer_edge_file, deezer_feature_file, deezer_edge_labels=None):
        deezer_edges = pd.read_csv(deezer_edge_file)
        deezer_feats = np.load(deezer_feature_file)

        adj_list = defaultdict(set)
        deezer_edges = deezer_edges.values.tolist()
        for e in deezer_edges:
            adj_list[int(e[0])].add(int(e[1]))
            adj_list[int(e[1])].add(int(e[0]))

        edge_labels = {}
        if deezer_edge_labels is not None:
            deezer_edge_labels = pd.read_csv(deezer_edge_labels)
            for i, r in deezer_edge_labels.iterrows():
                src = r["src"]
                dst = r["dst"]
                label = r["label"]

                edge_labels[(src, dst)] = label
                edge_labels[(dst, src)] = label

        return adj_list, deezer_feats, deezer_edges, edge_labels

    def get_twitch_data(self, twitch_edge_file, twitch_feature_file, twitch_edge_labels=None):
        feature_dim = 3170  # maximum feature id of all twitch datasets
        features_raw = json.load(open(twitch_feature_file))
        node2idx = {}
        features = []
        for i, it in enumerate(features_raw.items()):
            node2idx[int(it[0])] = i
            feats = np.zeros(feature_dim)
            feats[[int(val) for val in it[1]]] = 1
            features.append(feats)
        features = np.asarray([np.asarray(f) for f in features])
        edges_raw = pd.read_csv(twitch_edge_file)
        edges = edges_raw.values.tolist()
        edges = list(map(lambda x: [int(x[0]), int(x[1])], edges))
        adj_list = defaultdict(set)
        for e in edges:
            adj_list[node2idx[int(e[0])]].add(node2idx[int(e[1])])
            adj_list[node2idx[int(e[1])]].add(node2idx[int(e[0])])
        edge_labels = {}
        if twitch_edge_labels is not None:
            twitch_edge_labels = pd.read_csv(twitch_edge_labels)
            for i, r in twitch_edge_labels.iterrows():
                src = r["src"]
                dst = r["dst"]
                label = r["label"]

                edge_labels[(src, dst)] = label
                edge_labels[(dst, src)] = label

        return adj_list, features, edges, edge_labels

    def load_reddit_data(self, normalize=True, load_walks=False, prefix="reddit"):
        reddit_edges = pd.read_csv(self.config['file_path.reddit_edges'])
        reddit_feats = np.load(self.config['file_path.reddit_feats'])
        reddit_edge_labels = pd.read_csv(self.config['file_path.reddit_edge_labels'])

        adj_train = defaultdict(set)
        train_nodes = set()
        train_edges = []
        edge_labels = {}
        for i, r in reddit_edges.iterrows():
            e = r["edge"]
            src = int(e.split(",")[0][1:])
            dst = int(e.split(",")[1][1:-1])
            train_edges.append([src, dst])
            adj_train[src].add(dst)
            train_nodes.add(src)
            train_nodes.add(dst)

        for i, r in reddit_edge_labels.iterrows():
            src = r["src"]
            dst = r["dst"]
            label = r["label"]

            edge_labels[(src, dst)] = label

        train_nodes = list(train_nodes)

        if normalize and reddit_feats is not None:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(reddit_feats)
            reddit_feats = scaler.transform(reddit_feats)

        return adj_train, train_nodes, reddit_feats, train_edges, edge_labels
