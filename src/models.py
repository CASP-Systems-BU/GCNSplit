import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


class Classification(nn.Module):

    def __init__(self, emb_size, num_classes, device, hid_size=64):
        super(Classification, self).__init__()

        # self.weight = nn.Parameter(torch.FloatTensor(emb_size, num_classes))
        self.layer = nn.Sequential(
            nn.Linear(emb_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, num_classes)
        )
        self.init_params()
        self.device = device

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)

    def forward(self, embeds):
        logists = torch.softmax(self.layer(embeds), 1)
        return logists


class UnsupervisedLoss(object):
    """docstring for UnsupervisedLoss"""

    def __init__(self, adj_lists, train_nodes, device):
        super(UnsupervisedLoss, self).__init__()
        self.Q = 10
        self.N_WALKS = 6
        self.WALK_LEN = 3
        self.N_WALK_LEN = 3
        self.MARGIN = 3
        self.adj_lists = adj_lists
        self.train_nodes = train_nodes
        self.device = device

        self.target_nodes = None
        self.positive_pairs = []
        self.negtive_pairs = []
        self.node_positive_pairs = {}
        self.node_negtive_pairs = {}
        self.unique_nodes_batch = []

    def get_loss_sage(self, embeddings, nodes):
        assert len(embeddings) == len(self.unique_nodes_batch)
        assert False not in [nodes[i] == self.unique_nodes_batch[i] for i in range(len(nodes))]
        node2index = {n: i for i, n in enumerate(self.unique_nodes_batch)}

        nodes_score = []
        assert len(self.node_positive_pairs) == len(self.node_negtive_pairs)
        # print(len(self.node_positive_pairs))
        # print(len(self.node_negtive_pairs))
        for node in self.node_positive_pairs:
            pps = self.node_positive_pairs[node]
            nps = self.node_negtive_pairs[node]
            if len(pps) == 0 or len(nps) == 0:
                # print("PPS or NPS 0")
                # print(pps)
                # print(nps)
                continue

            # Q * Exception(negative score)
            indexs = [list(x) for x in zip(*nps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            neg_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
            neg_score = self.Q * torch.mean(torch.log(torch.sigmoid(-neg_score)), 0)
            # print(neg_score)

            # multiple positive score
            indexs = [list(x) for x in zip(*pps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            pos_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
            pos_score = torch.log(torch.sigmoid(pos_score))
            # print(pos_score)

            nodes_score.append(torch.mean(- pos_score - neg_score).view(1, -1))

        if len(nodes_score) == 0:
            print("NO NODES SCORE")
        loss = torch.mean(torch.cat(nodes_score, 0))

        return loss

    def extend_nodes(self, nodes, bfs=False, num_neg=6):
        self.positive_pairs = []
        self.node_positive_pairs = {}
        self.negtive_pairs = []
        self.node_negtive_pairs = {}

        self.target_nodes = nodes
        self.get_positive_nodes(nodes, bfs)
        # print(self.positive_pairs)
        self.get_negtive_nodes(nodes, num_neg)
        # print(self.negtive_pairs)
        self.unique_nodes_batch = list(
            set([i for x in self.positive_pairs for i in x]) | set([i for x in self.negtive_pairs for i in x]))
        # print("nodes: ", nodes)
        # print("unique nodes batch: ", self.unique_nodes_batch)
        # print("target nodes: ", self.target_nodes)
        assert set(self.target_nodes) <= set(self.unique_nodes_batch)
        return self.unique_nodes_batch

    def get_positive_nodes(self, nodes, bfs):
        if bfs:
            return self._run_bfs(nodes)
        return self._run_random_walks(nodes)

    def get_negtive_nodes(self, nodes, num_neg):
        for node in nodes:
            # if len(self.adj_lists[int(node)]) == 0:
            #     continue
            neighbors = set([node])
            frontier = set([node])
            for i in range(self.N_WALK_LEN):
                current = set()
                for outer in frontier:
                    current |= self.adj_lists[int(outer)]
                frontier = current - neighbors
                neighbors |= current
            far_nodes = set(self.train_nodes) - neighbors
            neg_samples = random.sample(far_nodes, num_neg) if num_neg < len(far_nodes) else far_nodes
            self.negtive_pairs.extend([(node, neg_node) for neg_node in neg_samples])
            self.node_negtive_pairs[node] = [(node, neg_node) for neg_node in neg_samples]
        return self.negtive_pairs

    def _run_bfs(self, nodes):
        if len(nodes) == 0:
            return self.positive_pairs
        queue = deque()
        for i in nodes:
            cur_pairs = []
            nbs = self.adj_lists[i]
            if len(nbs) == 0:
                self.positive_pairs.append((i, i))
                self.node_positive_pairs[i] = [(i, i)]
            else:
                for nb in nbs:
                    self.positive_pairs.append((i, nb))
                    cur_pairs.append((i, nb))
                self.node_positive_pairs[i] = cur_pairs
        return self.positive_pairs

    def _run_random_walks(self, nodes):
        for node in nodes:
            # if len(self.adj_lists[int(node)]) == 0:
            #     continue
            cur_pairs = []
            for i in range(self.N_WALKS):
                curr_node = node
                for j in range(self.WALK_LEN):
                    neighs = self.adj_lists[int(curr_node)]
                    if neighs:
                        next_node = random.choice(list(neighs))
                    else:
                        next_node = node
                    # self co-occurrences are useless
                    # my comment: in some cases there is no other option - when all nodes in batch have no outgoing
                    # edges this is the only way; thus I remove next_node != node condition
                    if next_node in self.train_nodes:
                        self.positive_pairs.append((node, next_node))
                        cur_pairs.append((node, next_node))
                    curr_node = next_node

            self.node_positive_pairs[node] = cur_pairs
        return self.positive_pairs


class SageLayer(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """

    def __init__(self, input_size, out_size, gcn=False):
        super(SageLayer, self).__init__()

        self.input_size = input_size
        self.out_size = out_size

        self.gcn = gcn
        self.weight = nn.Parameter(torch.FloatTensor(out_size, self.input_size if self.gcn else 2 * self.input_size))

        self.init_params()

    def init_params(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self, self_feats, aggregate_feats, neighs=None):
        """
        Generates embeddings for a batch of nodes.

        nodes	 -- list of nodes
        """
        # print("self: ", self_feats)
        # print("aggr: ", aggregate_feats)
        if not self.gcn:
            combined = torch.cat([self_feats, aggregate_feats], dim=1)
        else:
            combined = aggregate_feats
        combined = combined.float()
        combined = F.relu(self.weight.mm(combined.t())).t()
        return combined


class GraphSage(nn.Module):
    """docstring for GraphSage"""

    def __init__(self, num_layers, input_size, out_size, device, gcn=False, agg_func='MEAN'):
        super(GraphSage, self).__init__()

        self.input_size = input_size
        self.out_size = out_size
        self.num_layers = num_layers
        self.gcn = gcn
        self.device = device
        self.agg_func = agg_func

        for index in range(1, num_layers + 1):
            layer_size = out_size if index != 1 else input_size
            setattr(self, 'sage_layer' + str(index), SageLayer(layer_size, out_size, gcn=self.gcn))

    def forward(self, nodes_ids, nodes_features, adj_lists):
        """
        Generates embeddings for a batch of nodes.
        nodes_batch	-- batch of nodes to learn the embeddings
        """
        # print(len(nodes_ids))
        # print(len(nodes_features))
        # print(len(adj_lists))

        lower_layer_nodes = list(nodes_ids)
        nodes_batch_layers = [(lower_layer_nodes,)]
        # self.dc.logger.info('get_unique_neighs.')
        for i in range(self.num_layers):
            lower_samp_neighs, lower_layer_nodes_dict, lower_layer_nodes = self._get_unique_neighs_list(
                lower_layer_nodes, adj_lists)
            nodes_batch_layers.insert(0, (lower_layer_nodes, lower_samp_neighs, lower_layer_nodes_dict))

        assert len(nodes_batch_layers) == self.num_layers + 1
        pre_hidden_embs = nodes_features
        for index in range(1, self.num_layers + 1):
            # print("INDEX: ", index)
            nb = nodes_batch_layers[index][0]
            pre_neighs = nodes_batch_layers[index - 1]
            # self.dc.logger.info('aggregate_feats.')
            aggregate_feats = self.aggregate(nb, pre_hidden_embs, pre_neighs)
            sage_layer = getattr(self, 'sage_layer' + str(index))
            if index > 1:
                nb = self._nodes_map(nb, pre_neighs)
            # self.dc.logger.info('sage_layer.')
            # print(pre_hidden_embs)
            cur_hidden_embs = sage_layer(self_feats=pre_hidden_embs[nb],
                                         aggregate_feats=aggregate_feats)
            pre_hidden_embs = cur_hidden_embs

        return pre_hidden_embs

    def _nodes_map(self, nodes, neighs):
        layer_nodes, samp_neighs, layer_nodes_dict = neighs
        assert len(samp_neighs) == len(nodes)
        index = [layer_nodes_dict[x] for x in nodes]
        # print("nodes map: ", index)
        return index

    def _get_unique_neighs_list(self, nodes, adj_lists, num_sample=10):
        _set = set
        to_neighs = [adj_lists[int(node)] for node in nodes]
        if num_sample is not None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, num_sample)) if len(to_neigh) >= num_sample else to_neigh for to_neigh
                           in to_neighs]
        else:
            samp_neighs = to_neighs
        samp_neighs = [samp_neigh | set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        _unique_nodes_list = list(set.union(*samp_neighs))
        i = list(range(len(_unique_nodes_list)))
        unique_nodes = dict(list(zip(_unique_nodes_list, i)))
        return samp_neighs, unique_nodes, _unique_nodes_list

    def aggregate(self, nodes, pre_hidden_embs, pre_neighs, num_sample=10):
        unique_nodes_list, samp_neighs, unique_nodes = pre_neighs

        assert len(nodes) == len(samp_neighs)
        indicator = [(nodes[i] in samp_neighs[i]) for i in range(len(samp_neighs))]
        assert (False not in indicator)

        # Delete self loops
        # if not self.gcn:
        #     samp_neighs = [(samp_neighs[i] - set([nodes[i]])) for i in range(len(samp_neighs))]

        if len(pre_hidden_embs) == len(unique_nodes):
            embed_matrix = pre_hidden_embs
        else:
            embed_matrix = pre_hidden_embs[torch.LongTensor(unique_nodes_list)]
        # self.dc.logger.info('3')
        mask = torch.zeros(len(samp_neighs), len(unique_nodes))
        # print("samp neighs: ", len(samp_neighs))
        # print("unique nodes: ", len(unique_nodes))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        # self.dc.logger.info('4')

        if self.agg_func == 'MEAN':
            # print("mask ", mask)
            num_neigh = mask.sum(1, keepdim=True)
            mask = torch.div(mask, num_neigh).to(embed_matrix.device)
            # mask = mask.div(num_neigh).to(embed_matrix.device)
            # trick to remove NaNs in case num_neigh was 0
            mask[mask != mask] = 0
            aggregate_feats = mask.mm(embed_matrix)

        elif self.agg_func == 'MAX':
            mask[mask != mask] = 0
            indexs = [x.nonzero() for x in mask]
            aggregate_feats = []
            for feat in [embed_matrix[x.squeeze()] for x in indexs]:
                if len(feat.size()) == 1:
                    aggregate_feats.append(feat.view(1, -1))
                else:
                    aggregate_feats.append(torch.max(feat, 0)[0].view(1, -1))
            aggregate_feats = torch.cat(aggregate_feats, 0)

        # self.dc.logger.info('6')

        return aggregate_feats
