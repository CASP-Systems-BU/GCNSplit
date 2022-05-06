import argparse


def parser():
    parser = argparse.ArgumentParser(description='pytorch version of GraphSAGE')

    parser.add_argument('--dataset', type=str)
    parser.add_argument('--agg_func', type=str, default='MAX')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--sup_epochs', type=int, default=200)
    parser.add_argument('--b_sz', type=int, default=20)
    parser.add_argument('--sup_b_sz', type=int, default=0)
    parser.add_argument('--cut_coeff', type=float, default=1.0)
    parser.add_argument('--bal_coeff', type=float, default=0.00001)
    parser.add_argument('--gcn_coeff', type=float, default=1.0)
    parser.add_argument('--num_classes', type=int, default=0)
    parser.add_argument('--bfs', type=int, default=0)
    parser.add_argument('--graphsage_model', type=str, default="")
    parser.add_argument('--classification_model', type=str, default="")
    parser.add_argument('--lr', type=float, default=7.5e-5)
    parser.add_argument('--model', type=str, default="")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--gcn', action='store_true')
    parser.add_argument('--learn_method', type=str)
    parser.add_argument('--config', type=str, default='./src/experiments.conf')
    parser.add_argument('--num_steps', type=int)

    return parser
