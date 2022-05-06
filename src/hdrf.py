import sys
import time
from collections import defaultdict

import torch
from tqdm import tqdm


def partition_hdrf(edges, num_classes, load_imbalance, max_state_size=None):
    print("START HDRF PARTITIONING")
    print("MAX STATE SIZE: ", max_state_size)
    partial_degrees = defaultdict(lambda: 0)
    edge_partitions = {c: set() for c in range(num_classes)}
    vertex_partitions = {c: set() for c in range(num_classes)}
    max_size = 0
    min_size = 0
    processing_time = []
    idx = 0
    for src, dst in tqdm(edges):
        if idx == 0:
            start_time = time.time()
        if idx != 1 and idx % 10000 == 1:
            start_time = time.time()
        partial_degrees[src] += 1
        max_hdrf = float("-inf")
        max_p = None
        for ep, vp in zip(edge_partitions.items(), vertex_partitions.items()):
            cur_hdrf = hdrf(src, dst, ep[1], vp[1], partial_degrees, max_size, min_size, load_imbalance)
            if max_hdrf < hdrf(src, dst, ep[1], vp[1], partial_degrees, max_size, min_size, load_imbalance):
                max_hdrf = cur_hdrf
                max_p = ep[0]

        edge_partitions[max_p].add((src, dst))
        vertex_partitions[max_p].add(src)
        vertex_partitions[max_p].add(dst)

        temp_max_size = temp_min_size = None
        for ep in edge_partitions.values():
            size = len(ep)
            if temp_max_size is None or size > temp_max_size:
                temp_max_size = size
            if temp_min_size is None or size < temp_min_size:
                temp_min_size = size
        max_size = temp_max_size
        min_size = temp_min_size
        if idx > 0 and idx % 10000 == 0:
            end_time = time.time()
            processing_time.append([idx, end_time - start_time])
        if max_state_size is not None and get_size(vertex_partitions) + get_size(partial_degrees) > max_state_size:
            print("CLEAR!")
            vertex_partitions = {c: set() for c in range(num_classes)}
            partial_degrees.clear()
        idx += 1

    print("SIZE OF VERTEX PARTITIONS: ", get_size(vertex_partitions))
    print("SIZE OF EDGE PARTITIONS: ", get_size(edge_partitions))

    return edge_partitions, processing_time


def hdrf(v1, v2, ep, vp, partial_degrees, max_size, min_size, load_imbalance, eps=1):
    size = len(ep)

    return hdrf_rep(v1, v2, vp, partial_degrees) + load_imbalance * hdrf_bal(max_size, min_size, size, eps)


def hdrf_bal(max_size, min_size, size, eps=0.00001):
    # print("Load imbalance: {}".format(load_imbalance))
    result = (max_size - size) / (eps + max_size - min_size)
    return result


def hdrf_rep(v1, v2, vp, partial_degrees):
    g1 = hdrf_g(v1, normalize_degree(partial_degrees[v1], partial_degrees[v2]), vp)
    g2 = hdrf_g(v2, normalize_degree(partial_degrees[v2], partial_degrees[v1]), vp)
    return g1 + g2


def hdrf_g(v, norm_degree_v, vp):
    if v in vp:
        return 1 + 1 - norm_degree_v
    else:
        return 0


def normalize_degree(d1, d2):
    return d1 / (d1 + d2)


# Source: https://goshippo.com/blog/measure-real-size-any-python-object/
def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def hash_edge_partitioning(edges, name, num_classes):
    assignments = []
    cur = 0
    for e in tqdm(edges):
        assignments.append(cur)
        cur = (cur + 1) % num_classes
    return assignments


def evaluate_edge_partitioning(edges, partitioning, name, num_classes):
    nodes_assignments = defaultdict(set)
    freqs = [0] * num_classes
    # assert len(edges) == len(partitioning)
    for e, p in zip(edges, partitioning):
        if isinstance(p, torch.Tensor):
            p = p.item()
        freqs[p] += 1
        if e[0] not in nodes_assignments:
            nodes_assignments[e[0]].add(p)
        elif p not in nodes_assignments[e[0]]:
            nodes_assignments[e[0]].add(p)
        if e[1] not in nodes_assignments:
            nodes_assignments[e[1]].add(p)
        elif p not in nodes_assignments[e[1]]:
            nodes_assignments[e[1]].add(p)
    vertex_copies = sum([len(copies) for copies in nodes_assignments.values()])
    print("FREQS of {}: {}".format(name, freqs))
    normalized_load = max(freqs) / (len(partitioning) // num_classes)
    print("NORMALIZED LOAD of {}: {}".format(name, normalized_load))
    replication_factor = vertex_copies / len(nodes_assignments)
    print("REPLICATION FACTOR of {}: {}".format(name, replication_factor))
    return normalized_load, replication_factor
