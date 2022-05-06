import multiprocessing
import time
from queue import Empty
import os

import torch


def partition_batches_from_queue(reading_queue: multiprocessing.Queue, writing_queue: multiprocessing.Queue, adj_list,
                                 bidirectional, features, gap, graphsage, args):
    freqs = [0] * args.num_classes
    start_time = time.time()
    pid = os.getpid()

    while True:
        try:
            (edges_batch, features_batch, node2idx) = reading_queue.get(block=True, timeout=10)
            if features_batch is None:
                features_batch = features
        except Empty:
            print("Timeout during reading from queue.")
            finish_time = time.time()
            print("TIME: ", finish_time - start_time)
            return

        if node2idx is None:
            assignments, processing_time = _partition_edges_batch(adj_list, edges_batch, bidirectional, features_batch,
                                                                  freqs, gap,
                                                                  graphsage,
                                                                  args.learn_method, args.num_classes, args.max_load,
                                                                  args.sorted_inference)
        else:
            assignments, processing_time = _partition_batch(adj_list, edges_batch, bidirectional,
                                                            features_batch, freqs,
                                                            gap, graphsage,
                                                            args.learn_method, args.num_classes,
                                                            args.max_load,
                                                            args.sorted_inference, node2idx)

        try:
            writing_queue.put((assignments, processing_time, pid),  block=True, timeout=10)
        except Empty:
            print("Timeout during writing to WRITING queue!")
            return


def _partition_batch(adj_list, edges_batch, bidirectional, features_batch, freqs, gap, graphsage,
                     learn_method,
                     num_partitions, max_load, sorted_inference, node2idx):
    added_edges = set()
    assignments = []
    num_assignments = sum([p for p in freqs])
    for e in edges_batch:
        if node2idx[e[0]][0] not in adj_list:
            adj_list[node2idx[e[0]][0]].add(node2idx[e[1]][0])
            added_edges.add((node2idx[e[0]][0], node2idx[e[1]][0]))
        if bidirectional and node2idx[e[1][0]] not in adj_list[node2idx[e[0]][0]]:
            adj_list[node2idx[e[1]][0]].add(node2idx[e[0][0]])
            added_edges.add((node2idx[e[1]][0], node2idx[e[0]][0]))
    with torch.no_grad():
        if learn_method == "sup_edge":
            # list of all source nodes in the batch
            srcs = [item for item in map(lambda e: e[0], edges_batch)]
            # list of all destination nodes in the batch
            dsts = [item for item in map(lambda e: e[1], edges_batch)]

            srcs2idx = [item for item in map(lambda e: node2idx[e[0]][0], edges_batch)]
            dsts2idx = [item for item in map(lambda e: node2idx[e[1]][0], edges_batch)]

            srcs_embs = graphsage(srcs2idx, features_batch, adj_list)
            dsts_embs = graphsage(dsts2idx, features_batch, adj_list)
            embs = torch.cat([srcs_embs, dsts_embs], 1)
            predicts = gap(embs)
            _, max_idx = torch.max(predicts, dim=1)
            max_idx = max_idx.squeeze().tolist()
            n = 0
            while n < len(srcs_embs):
                num_assignments += 1
                perfect_load = num_assignments / num_partitions
                if sorted_inference:
                    p = _get_sorted_edge_inference(freqs, max_load, perfect_load, predicts[n])
                else:
                    p = _get_least_loaded_edge_inference(freqs, max_load, perfect_load, max_idx[n])
                assignments.append([srcs[n], dsts[n], p])
                n += 1
        else:
            pairs = [item for sublist in map(lambda e: [e[0], e[1]], edges_batch) for item in
                     sublist]
            pairs2idx = [item for sublist in map(lambda e: [node2idx[e[0]][0], node2idx[e[1]][0]], edges_batch) for item
                         in
                         sublist]
            embs = graphsage(pairs2idx, features_batch, adj_list)
            predicts = gap(embs)
            max_val, max_idx = torch.max(predicts, dim=1)
            max_val = max_val.squeeze().tolist()
            max_idx = max_idx.squeeze().tolist()

            n = 0
            while n < len(pairs2idx) - 1:
                num_assignments += 1
                perfect_load = num_assignments / num_partitions
                if sorted_inference:
                    p = _get_sorted_inference(freqs, max_load, perfect_load, predicts[n:n + 2])
                else:
                    p = _get_least_loaded_inference(freqs, max_load, perfect_load, max_val[n:n + 2],
                                                    max_idx[n:n + 2])
                assignments.append([pairs[n], pairs[n + 1], p])
                n += 2
    for e in added_edges:
        adj_list[e[0]].discard(e[1])
    added_edges.clear()
    return assignments, time.time()


def _partition_edges_batch(adj_list, edges_batch, bidirectional, features_batch, freqs, gap, graphsage, learn_method,
                           num_partitions, max_load, sorted_inference):
    added_edges = set()
    assignments = []
    num_assignments = sum([p for p in freqs])
    for e in edges_batch:
        if e[0] not in adj_list:
            adj_list[e[0]].add(e[1])
            added_edges.add((e[0], e[1]))
        if bidirectional and e[1] not in adj_list[e[0]]:
            adj_list[e[1]].add(e[0])
            added_edges.add((e[1], e[0]))
    with torch.no_grad():
        if learn_method == "sup_edge":
            srcs = [item for item in map(lambda e: e[0], edges_batch)]
            dsts = [item for item in map(lambda e: e[1], edges_batch)]
            srcs_embs = graphsage(srcs, features_batch, adj_list)
            dsts_embs = graphsage(dsts, features_batch, adj_list)
            embs = torch.cat([srcs_embs, dsts_embs], 1)
            predicts = gap(embs)
            _, max_idx = torch.max(predicts, dim=1)
            max_idx = max_idx.squeeze().tolist()
            n = 0
            while n < len(srcs_embs):
                num_assignments += 1
                perfect_load = num_assignments / num_partitions
                if sorted_inference:
                    p = _get_sorted_edge_inference(freqs, max_load, perfect_load, predicts[n])
                else:
                    p = _get_least_loaded_edge_inference(freqs, max_load, perfect_load, max_idx[n])
                assignments.append([srcs[n], dsts[n], p])
                n += 1
        else:
            pairs = [item for sublist in map(lambda e: [e[0], e[1]], edges_batch) for item in sublist]
            embs = graphsage(pairs, features_batch, adj_list)
            predicts = gap(embs)
            max_val, max_idx = torch.max(predicts, dim=1)
            max_val = max_val.squeeze().tolist()
            max_idx = max_idx.squeeze().tolist()

            n = 0
            while n < len(pairs) - 1:
                num_assignments += 1
                perfect_load = num_assignments / num_partitions
                if sorted_inference:
                    p = _get_sorted_inference(freqs, max_load, perfect_load, predicts[n:n + 2])
                else:
                    p = _get_least_loaded_inference(freqs, max_load, perfect_load, max_val[n:n + 2],
                                                    max_idx[n:n + 2])
                assignments.append([pairs[n], pairs[n + 1], p])
                n += 2
    for e in added_edges:
        adj_list[e[0]].discard(e[1])
    added_edges.clear()
    return assignments, time.time()


def _get_sorted_inference(freqs, max_load, perfect_load, predicts):
    sorted_values, sorted_partitions = torch.sort(predicts, descending=True)
    for e1, e2 in zip(sorted_partitions[0].flatten().split(1), sorted_partitions[1].flatten().split(1)):
        e1, e2 = e1.item(), e2.item()
        if predicts[0][e1] > predicts[1][e2]:
            p = e1
        else:
            p = e2
        freqs[p] += 1
        if get_load_value(freqs, perfect_load) < max_load:
            return p
        else:
            freqs[p] -= 1
            p = e1 if p == e2 else e2
            freqs[p] += 1
            if get_load_value(freqs, perfect_load) < max_load:
                return p
            freqs[p] -= 1
    p = freqs.index(min(freqs))
    freqs[p] += 1
    return p


def _get_least_loaded_inference(freqs, max_load, perfect_load, max_val, max_idx):
    v1, v2 = max_val[0], max_val[1]
    i1, i2 = max_idx[0], max_idx[1]
    if i1 != i2:
        if v1 > v2:
            # i1 = i1.item()
            p = i1
        else:
            # i2 = i2.item()
            p = i2
        freqs[p] += 1
        if get_load_value(freqs, perfect_load) < max_load:
            return p
        else:
            freqs[p] -= 1
            # p = i1.item() if p == i2 else i2.item()
            p = i1 if p == i2 else i2
            freqs[p] += 1
            if get_load_value(freqs, perfect_load) < max_load:
                return p
            freqs[p] -= 1
    else:
        p = i1
        freqs[p] += 1
        if get_load_value(freqs, perfect_load) < max_load:
            return p
        else:
            freqs[p] -= 1
    p = freqs.index(min(freqs))
    freqs[p] += 1
    return p


def _get_sorted_edge_inference(freqs, max_load, perfect_load, predicts):
    sorted_values, sorted_partitions = torch.sort(predicts, descending=True)
    for p in sorted_partitions.flatten().split(1):
        p = p.item()
        freqs[p] += 1
        if get_load_value(freqs, perfect_load) < max_load:
            return p
        else:
            freqs[p] -= 1
    p = freqs.index(min(freqs))
    freqs[p] += 1
    return p


def _get_least_loaded_edge_inference(freqs, max_load, perfect_load, max_idx):
    freqs[max_idx] += 1
    if get_load_value(freqs, perfect_load) < max_load:
        return max_idx
    else:
        freqs[max_idx] -= 1
    p = freqs.index(min(freqs))
    freqs[p] += 1
    return p


def get_load_value(freqs, perfect_load):
    return max(freqs) / perfect_load
