import argparse
import copy
from collections import defaultdict
from datetime import datetime

import pyhocon
import torch
from torch import multiprocessing
from torch.multiprocessing import Queue, Process

from src.dataCenter import DataCenter
from src.stream_partitioner import partition_batches_from_queue
from src.stream_reader import read_edge_batches_with_features_to_queue, read_edge_batches_to_queue
from src.stream_writer import write_batches_from_queue_to_file

if __name__ == '__main__':
    print("RUN STREAM PARTITIONING")
    parser = argparse.ArgumentParser(description='Streaming partitioning with GCN')

    parser.add_argument('--learn_method', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--num_classes', type=int)
    parser.add_argument('--max_load', type=float)
    parser.add_argument('--model', type=str)
    parser.add_argument('--inf_b_sz', type=int)
    parser.add_argument('--num_processes', type=int)
    parser.add_argument('--edge_file_path', type=str)
    parser.add_argument('--stream_features', action='store_true')
    parser.add_argument('--feats_file_path', type=str)
    parser.add_argument('--sorted_inference', action='store_true')
    parser.add_argument('--with_train_adj', action='store_true')
    parser.add_argument('--throughput_report', type=int, default=1)
    parser.add_argument('--config', type=str, default='./experiments.conf')

    args = parser.parse_args()

    print(args)

    map_location = 'cpu'
    device = "cpu"
    # device = torch.device("cpu")
    print('DEVICE:', device)

    # load config file
    config = pyhocon.ConfigFactory.parse_file(args.config)

    # load data
    ds = args.dataset
    dataCenter = DataCenter(config)

    if args.stream_features:
        features = None
    else:
        features_ndarray = dataCenter.get_train_features(ds, args.feats_file_path)
        features = torch.from_numpy(features_ndarray).to(device)

    if args.with_train_adj:
        adj_list_train = dataCenter.get_train_adj_list(ds)
    else:
        adj_list_train = defaultdict(set)

    if args.model != "":
        [graphSage, classification] = torch.load(args.model, map_location=map_location)
    elif args.graphsage_model != "" and args.classification_model != "":
        graphSage = torch.load(args.graphsage_model, map_location=map_location)
        classification = torch.load(args.classification_model, map_location=map_location)
    else:
        raise Exception("You have to specify a model to run!")

    graphSage.to(device)

    if args.num_classes == 0:
        num_labels = len(set(getattr(dataCenter, ds + '_labels')))
    else:
        num_labels = args.num_classes
    classification.to(device)

    graphSage.share_memory()
    classification.share_memory()
    multiprocessing.set_start_method("spawn", force=True)

    pool = multiprocessing.Pool()
    reading_queue = Queue(maxsize=100)
    if args.stream_features:
        reader = Process(target=read_edge_batches_with_features_to_queue,
                         args=(args.feats_file_path, args.inf_b_sz, reading_queue))
    else:
        reader = Process(target=read_edge_batches_to_queue,
                         args=(args.edge_file_path, args.inf_b_sz, reading_queue))
    reader.start()

    processes = []
    writing_queue = Queue(maxsize=100)
    for i in range(args.num_processes):
        processes.append(Process(target=partition_batches_from_queue,
                                 args=(reading_queue, writing_queue, copy.deepcopy(adj_list_train), False, features,
                                       classification, graphSage, args)))

    for p in processes:
        p.start()

    model_path = args.model.split("/")[-1]

    filename = "ds-{}-{}_win-size-{}_{}_{}_RESULTS.csv".format(
        args.dataset,
        args.inf_b_sz,
        model_path,
        args.num_classes,
        datetime.now().strftime('%b%d_%H-%M-%S'))

    writer = Process(target=write_batches_from_queue_to_file, args=(writing_queue, filename, args.throughput_report))
    writer.start()

    reader.join()
    for p in processes:
        p.join()
    writer.join()
