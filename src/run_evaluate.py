import argparse
from collections import defaultdict

import pandas as pd

parser = argparse.ArgumentParser(description='HDRF python implementation')

parser.add_argument('--input_file', type=str, default='')
parser.add_argument('--num_classes', type=int)
parser.add_argument('--chunk_size', type=int, default=100000)

args = parser.parse_args()

def parse_one_batch(edges_assignments, nodes_assignments, freqs):
    # assert len(edges) == len(partitioning)
    for r in edges_assignments:
        src = r[0]
        dst = r[1]
        p = r[2]
        freqs[p] += 1
        if src not in nodes_assignments:
            nodes_assignments[src].add(p)
        elif p not in nodes_assignments[src]:
            nodes_assignments[src].add(p)
        if dst not in nodes_assignments:
            nodes_assignments[dst].add(p)
        elif p not in nodes_assignments[dst]:
            nodes_assignments[dst].add(p)


nodes_assignments = defaultdict(set)
freqs = [0] * args.num_classes
edges_number = 0
for batch in pd.read_csv(args.input_file, chunksize=args.chunk_size):
    edges_assignments = batch.values.tolist()
    edges_number += len(edges_assignments)
    parse_one_batch(edges_assignments, nodes_assignments, freqs)

vertex_copies = sum([len(copies) for copies in nodes_assignments.values()])
print("FREQS: {}".format(freqs))
normalized_load = max(freqs) / (edges_number // args.num_classes)
print("NORMALIZED LOAD: {}".format(normalized_load))
replication_factor = vertex_copies / len(nodes_assignments)
print("REPLICATION FACTOR: {}".format(replication_factor))