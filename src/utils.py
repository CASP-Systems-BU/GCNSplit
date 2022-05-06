import math
import sys

import numpy as np
import torch
from sklearn.utils import shuffle


# b_sz=0 means to take the whole dataset in each epoch
def train_supervised_edge_partitioning(train_edges, train_features, graphSage, classification, train_adj_list,
                                       num_classes, labels, cut_coeff, bal_coeff, device, tensorboard, b_sz, epochs,
                                       lr):
    models = [graphSage, classification]
    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                params.append(param)

    optimizer = torch.optim.Adam(params, lr=lr)
    for model in models:
        model.zero_grad()
    for epoch in range(epochs):
        train_edges = shuffle(train_edges)
        if b_sz <= 0:
            b_sz = len(train_edges)
        batches = math.ceil(len(train_edges) / b_sz)
        visited_edges = 0
        for index in range(batches):
            edges_batch = train_edges[index * b_sz:(index + 1) * b_sz]
            nodes_batch = [n for e in edges_batch for n in [e[0], e[1]]]
            visited_edges += len(edges_batch)
            embs_batch = graphSage(nodes_batch, train_features, train_adj_list)
            supervised_loss = get_supervised_partitioning_loss(train_adj_list, classification, embs_batch, nodes_batch,
                                                               edges_batch,
                                                               num_classes, labels, cut_coeff, bal_coeff, device,
                                                               tensorboard, epoch,
                                                               index + 1, batches)

            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Dealed Edges [{}/{}] '.format(epoch + 1, epochs, index,
                                                                                            batches,
                                                                                            supervised_loss.item(),
                                                                                            visited_edges,
                                                                                            len(train_edges)))

            global_step = epoch * batches + index + 1
            tensorboard.add_scalar("supervised_loss", supervised_loss.item(), global_step=global_step)
            supervised_loss.backward()

            # nn.utils.clip_grad_norm_(classification.parameters(), 5)
            optimizer.step()
            optimizer.zero_grad()
            for model in models:
                model.zero_grad()

    return classification, graphSage


def get_gap_loss(adj_list, node_bal_coeff, classification, cut_coeff, embs_batch, nodes_batch, num_classes, device,
                 tensorboard, epoch=-1, step=-1, num_steps=-1):
    logists = classification(embs_batch)
    node2index = {n: i for i, n in enumerate(nodes_batch)}
    # print("NODES BATCH:", nodes_batch)
    batch_adj_list = {}
    for node in nodes_batch:
        batch_adj_list[int(node)] = []
    for node in nodes_batch:
        batch_adj_list[int(node)] = [n for n in adj_list[int(node)] if n in batch_adj_list]
    # print("batch_adj_list: ", batch_adj_list)
    D = torch.tensor([len(v) for v in batch_adj_list.values()], dtype=torch.float).to(device)
    D.requires_grad = False
    # print("D: ", D)
    A = [[0 for i in range(len(nodes_batch))] for j in range(len(nodes_batch))]
    # print(A)
    for node, nbs in batch_adj_list.items():
        for nb in nbs:
            A[node2index[node]][node2index[nb]] = 1
    # print(A)
    A = torch.tensor(A, dtype=torch.float, device=device)
    A.to(device)
    # print(torch.sum(A, 1))
    assert (torch.sum(A, 1).equal(D))
    A.requires_grad = False
    gamma = logists.T @ D
    gamma.to(device)

    y_div_gamma = logists / gamma
    y_div_gamma.to(device)
    # print("y/gamma: ", y_div_gamma)
    # print("y/gamma[0]: ", y_div_gamma[0])
    one_minus_Y_T = (1 - logists).T
    one_minus_Y_T.to(device)
    mm = y_div_gamma @ one_minus_Y_T
    mm.to(device)
    # print("MM: ", mm)
    times_adj_matrix = mm * A
    times_adj_matrix.to(device)
    # print("TIMES A: ", times_adj_matrix)
    left_sum = times_adj_matrix.sum()
    left_sum.to(device)
    # print("LEFT SUM: ", left_sum)
    num_nodes = len(nodes_batch)
    cluster_size = torch.sum(logists, dim=0).to(device)
    ground_truth = torch.tensor([num_nodes / float(num_classes)] * num_classes).to(device)
    mse_loss = torch.nn.modules.loss.MSELoss()
    node_bal = mse_loss(ground_truth, cluster_size)
    node_bal.to(device)

    # print("Bal: ", bal)
    # / len(nodes_batch) makes it the same regardless of window size
    # loss = (cut_coeff * left_sum + bal_coeff * bal) / len(nodes_batch)
    loss = (cut_coeff * left_sum + node_bal_coeff * node_bal)

    if step != -1:
        global_step = epoch * num_steps + step
        tensorboard.add_scalar("edge_cut", left_sum.item(), global_step=global_step)
        tensorboard.add_scalar("node_balance", node_bal.item(), global_step=global_step)
        tensorboard.add_scalar("node_gap_loss", loss.item(), global_step=global_step)
    return loss


def get_supervised_partitioning_loss(adj_list, classification, embs_batch, nodes_batch, edges_batch, num_classes,
                                     labels, cut_coeff,
                                     bal_coeff, device, tensorboard, epoch, step, num_steps):
    node2index = {n: i for i, n in enumerate(nodes_batch)}
    labels = torch.tensor([labels[(e[0], e[1])] for e in edges_batch]).to(device)
    edges_embeddings = []
    for src, dst in edges_batch:
        src_emb = embs_batch[node2index[src]]
        dst_emb = embs_batch[node2index[dst]]
        # print("SRC EMB SIZE: ", src_emb.size())
        edge_emb = torch.cat([src_emb, dst_emb], 0)
        # print("EDGE EMB SIZE: ", edge_emb.size())
        edges_embeddings.append(edge_emb)
    edges_embeddings = torch.stack(edges_embeddings)
    logits = classification(edges_embeddings).to(device)
    num_edges = len(edges_batch)
    cluster_size = torch.sum(logits, dim=0).to(device)
    ground_truth = torch.tensor([num_edges / float(num_classes)] * num_classes).to(device)
    mse_loss = torch.nn.modules.loss.MSELoss()
    bal = mse_loss(ground_truth, cluster_size)
    partitioning = torch.nn.CrossEntropyLoss()
    partitioning_loss = partitioning(logits, labels)
    if step != -1:
        global_step = epoch * num_steps + step
        tensorboard.add_scalar("edge_balance", bal.item(), global_step=global_step)
        tensorboard.add_scalar("classification", partitioning_loss.item(), global_step=global_step)

    return cut_coeff * partitioning(logits, labels) + bal_coeff * bal


def apply_model(nodes, features, graphSage, classification, unsupervised_loss, adj_list, args, epoch, tensorboard,
                device, num_steps=None):
    num_neg = 100
    train_nodes = shuffle(nodes)

    models = [graphSage, classification]
    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                params.append(param)

    optimizer = torch.optim.Adam(params, lr=7.5e-5)
    optimizer.zero_grad()
    for model in models:
        model.zero_grad()

    if args.b_sz == -1:
        batches = 1
    else:
        batches = math.ceil(len(train_nodes) / args.b_sz)
    visited_nodes = set()
    if num_steps is None:
        num_steps = batches
    for index in range(num_steps):
        if args.b_sz == -1:
            nodes_batch = train_nodes
        else:
            nodes_batch = train_nodes[index * args.b_sz:(index + 1) * args.b_sz]

        # extend nodes batch for unspervised learning
        # no conflicts with supervised learning
        # print("not extended: ", len(nodes_batch))
        extended_nodes_batch = np.asarray(
            list(unsupervised_loss.extend_nodes(nodes_batch, num_neg=num_neg, bfs=args.bfs)))
        visited_nodes |= set(extended_nodes_batch)

        # get ground-truth for the nodes batch
        # labels_batch = labels[nodes_batch]

        # feed nodes batch to the graphSAGE
        # returning the nodes embeddings
        extended_embs_batch = graphSage(extended_nodes_batch, features, adj_list)
        embs_batch = extended_embs_batch
        print("NODES BATCH LEN: ", len(nodes_batch))
        nodes_batch = extended_nodes_batch
        print("EXTENDED NODES BATCH LEN: ", len(nodes_batch))
        if args.learn_method == 'gap':
            loss = get_gap_loss(adj_list, args.bal_coeff, classification, args.cut_coeff, embs_batch, nodes_batch,
                                args.num_classes, device=device, epoch=epoch, step=index + 1, num_steps=num_steps,
                                tensorboard=tensorboard)
        elif args.learn_method == 'gap_plus':
            gap_loss = get_gap_loss(adj_list, args.bal_coeff, classification, args.cut_coeff, embs_batch, nodes_batch,
                                    args.num_classes, device=device, epoch=epoch, step=index + 1, num_steps=num_steps,
                                    tensorboard=tensorboard)

            loss_net = unsupervised_loss.get_loss_sage(extended_embs_batch, extended_nodes_batch)
            tensorboard.add_scalar("GCN loss", loss_net.item(), global_step=epoch * num_steps + index + 1)
            loss = args.gcn_coeff * loss_net + gap_loss
        else:
            loss_net = unsupervised_loss.get_loss_sage(extended_embs_batch, extended_nodes_batch)
            tensorboard.add_scalar("GCN loss", loss_net.item(), global_step=epoch * num_steps + index + 1)
            loss = loss_net

        print('Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(index + 1, num_steps, loss.item(),
                                                                         len(visited_nodes), len(train_nodes)))

        loss.backward()
        optimizer.step()

        optimizer.zero_grad()
        for model in models:
            model.zero_grad()

    return graphSage, classification
