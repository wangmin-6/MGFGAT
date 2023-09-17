import argparse
from itertools import product
from multiprocessing.dummy import freeze_support

import numpy as np

import torch
import multiprocessing
import timeit
import operator
# from same_baseline import (GCN,GIN,GraphSAGE,GAT)
from source_baseline import (GCN,GIN,GraphSAGE,GAT)
from util_thread import fun


def get():
    nums = [
        ["AD", "CN", [10, 4], [GIN,5, 64]],
        ["EMCI", "LMCI", [8, 10], [GIN, 5, 64]],
        ["MCI", "CN", [5, 5], [GIN, 5, 64]]
    ]
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)  # y
    parser.add_argument('--batch_size', type=int, default=32)  # y
    parser.add_argument('--lr', type=float, default=0.01)  # y
    parser.add_argument('--flag', type=str, default="-source")
    parser.add_argument('--is_lr_decay_factor', type=bool, default=False)
    parser.add_argument('--lr_decay_factor', type=float, default=0.5)  # 默认
    parser.add_argument('--lr_decay_step_size', type=int, default=50)  # 默认
    parser.add_argument('--weight_decay', type=float, default=0.0)  # 默认
    args = parser.parse_args()
    return nums, args, device


def run(num_roi):
    nums, args, device = get()
    print(device)
    # for Net, num_layers, hidden in net_num:
    for class1, class2, resample_num, net_num in nums:
        # fusion_orgin = "fusion"
        Net = net_num[0]
        num_layers = net_num[1]
        hidden = net_num[2]
        fusion_orgin = "orgin"
        accs = []
        if fusion_orgin == "fusion":
            log = "data/log/result/"
            filename = "baseline fusion.csv"
            map_root = "data/imageid/resample/fusion_roi/" + class1 + "_" + class2 + " ex_split_map 10.pt"
            data_root = "data/imageid/resample/fusion_roi/" + class1 + "_" + class2 + " roi " + str(num_roi) + ".pt"
            acc = fun(Net, num_layers, hidden, args, class1, class2, resample_num, num_roi, log, data_root, map_root,
                device, filename)
            accs.append(acc)
        else:
            filename = "baseline source.csv"
            log = "data/log/result orgin/"
            map_root = "data/imageid/resample/orgin_roi/" + class1 + "_" + class2 + " ex_split_map 10.pt"
            data_root = "data/imageid/resample/orgin_roi/" + class1 + "_" + class2 + " roi " + str(num_roi) + ".pt"
            acc = fun(Net, num_layers, hidden, args, class1, class2, resample_num, num_roi, log, data_root, map_root,
                device, filename)
            accs.append(acc)



if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    freeze_support()
    num_rois = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    p = multiprocessing.Pool(5)
    b = p.map(run, num_rois)

