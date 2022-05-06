import argparse

import pandas as pd
from tqdm import tqdm

from src.hdrf import partition_hdrf, evaluate_edge_partitioning, hash_edge_partitioning

parser = argparse.ArgumentParser(description='HDRF python implementation')

parser.add_argument('--input_file', type=str, default='')
parser.add_argument('--num_classes', type=int)
parser.add_argument('--load_imbalance', type=float, default=1)
parser.add_argument('--output_file', type=str, default='')
parser.add_argument('--max_state_size', type=int)

args = parser.parse_args()

print(args)


def parse_reddit_edges(edges):
    parsed = []
    for i, r in tqdm(edges.iterrows()):
        e = r["edge"]
        src = int(e.split(",")[0][1:])
        dst = int(e.split(",")[1][1:-1])
        # yield [src, dst]
        parsed.append([src, dst])
    return parsed


def parse_csv_edges(edges):
    edges = edges.values.tolist()
    edges = list(map(lambda x: [int(x[0]), int(x[1])], edges))
    return edges


if "reddit" in args.input_file:
    edges = parse_reddit_edges(pd.read_csv(args.input_file))
else:
    edges = parse_csv_edges(pd.read_csv(args.input_file))

partitions, processing_time = partition_hdrf(edges, args.num_classes, args.load_imbalance, args.max_state_size)

result = []
for label, labeled_edges in partitions.items():
    for e in labeled_edges:
        result.append({"src": e[0], "dst": e[1], "label": label})

assigned_edges = pd.DataFrame(result)
if args.output_file == "":
    output_file = args.input_file + "_labeled_" + str(args.num_classes)
else:
    output_file = args.output_file
assigned_edges.to_csv(output_file, index=False)
processing_time = pd.DataFrame(processing_time)
processing_time.to_csv(output_file + "_processing_time")

evaluate_edge_partitioning([e for s in partitions.values() for e in s], [p for p in partitions for _ in partitions[p]],
                           "HDRF", args.num_classes)

hash_partitions = hash_edge_partitioning(edges, "HASH", args.num_classes)
evaluate_edge_partitioning(edges, hash_partitions, "HASH", args.num_classes)
