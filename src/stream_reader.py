from queue import Full

import pandas as pd
import numpy as np
import torch
import os
from torch.multiprocessing import Queue


def read_edge_batches_with_features_to_queue(data_path, batch_size, queue: Queue):
    i = 0
    files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(data_path) for f in filenames if
             os.path.splitext(f)[1] == '.csv']
    for file in files:
        for batch in pd.read_csv(file, chunksize=batch_size):
            batch = np.array(batch)
            edges = batch[:, 0:2]
            features_unprocessed = batch[:, 2:]
            features = []
            feature_length = int(len(features_unprocessed[0]) / 2)
            for feature in features_unprocessed:
                f_src = feature[0:feature_length]
                f_dst = feature[feature_length:]
                features.append(f_src)
                features.append(f_dst)
            node2idx = {}
            current = 0

            print(i)
            i += batch_size
            # reddit dataset processing
            if len(edges) > 0:
                for edge in edges:
                    if edge[0] not in node2idx:
                        node2idx.setdefault(edge[0], []).append(current)
                        current += 1
                    if edge[1] not in node2idx:
                        node2idx.setdefault(edge[1], []).append(current)
                        current += 1
                elements = (edges, torch.tensor(features), node2idx)
                try:
                    queue.put(elements, block=True)
                except Full:
                    print("Timeout during writing to READING queue.")
                    return


print("End of files to read.")


def read_edge_batches_to_queue(file_path, batch_size, queue: Queue):
    i = 0
    for batch in pd.read_csv(file_path, chunksize=batch_size):
        print(i)
        i += batch_size
        edges_list = batch.values.tolist()
        # reddit dataset processing
        if "reddit" in file_path:
            edges = list(map(lambda x: [int(x[1].split(",")[0][1:]), int(x[1].split(",")[1][1:-1])], edges_list))
        else:
            edges = edges_list
        if len(edges) > 0:
            try:
                queue.put((edges, None, None), block=True)
            except Full:
                print("Timeout during writing to READING queue.")
                return
    print("End of file to read.")
