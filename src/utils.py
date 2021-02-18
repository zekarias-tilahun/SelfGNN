from torch_geometric.datasets import Planetoid, Coauthor, Amazon, CitationFull, Flickr, Yelp
from torch_geometric.transforms import GDC, LocalDegreeProfile
from torch_geometric.utils import to_scipy_sparse_matrix

from collections import Counter
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import subprocess
import argparse
import os.path as osp
import os

import scipy.sparse as sp
import numpy as np
import torch


class Augmentations:

    def __init__(self, method='gdc'):
        methods = {"ppr", "heat", "ldp", "paste", "split", "zscore", "katz"}
        assert method in methods
        self.method = method

    @staticmethod
    def _split(data, permute=True):
        perm = torch.randperm(
            data.x.shape[1]) if permute else torch.arange(data.x.shape[1])
        x = data.x.clone()
        x = x[:, perm]
        size = x.shape[1] // 2
        x1 = x[:, :size]
        x2 = x[:, size:]
        new_data = data.clone()
        data.x = x1
        new_data.x = x2
        return new_data

    @staticmethod
    def _standardize(data):
        x = data.x
        mean, std = x.mean(dim=0), x.std(dim=0)
        new_data = data.clone()
        new_data.x = (x - mean) / (std + 10e-7)
        return new_data

    @staticmethod
    def _katz(data, beta=0.1, threshold=0.0001):
        adj_matrix = to_scipy_sparse_matrix(data.edge_index)
        num_nodes = adj_matrix.shape[0]
        a_hat = adj_matrix + sp.eye(num_nodes)
        d_hat = sp.diags(
            np.array(1 / np.sqrt(a_hat.sum(axis=1))).reshape(num_nodes))
        a_hat = d_hat @ a_hat @ d_hat
        temp = sp.eye(num_nodes) - beta * a_hat
        h_katz = (sp.linalg.inv(temp.tocsc()) * beta * a_hat).toarray()
        mask = (h_katz < threshold)
        h_katz[mask] = 0.
        edge_index = torch.tensor(h_katz.nonzero(), dtype=torch.long)
        edge_attr = torch.tensor(h_katz[h_katz.nonzero()], dtype=torch.float32)
        new_data = data.clone()
        new_data.edge_index = edge_index
        new_data.edge_attr = edge_attr
        return new_data

    def __call__(self, data):

        if self.method == 'ppr':
            return GDC(diffusion_kwargs={'alpha': 0.15, 'method': 'ppr'})(data.clone())
        elif self.method == 'heat':
            return GDC(diffusion_kwargs={'t': 3, 'method': 'heat'})(data.clone())
        elif self.method == "katz":
            return self._katz(data)
        elif self.method == 'paste':
            return LocalDegreeProfile()(data.clone())
        elif self.method == 'ldp':
            d = data.clone()
            d.x = None
            return LocalDegreeProfile()(d)
        elif self.method == 'split':
            return self._split(data)
        elif self.method == "zscore":
            return self._standardize(data)

    def __str__(self):
        if self.method == "ppr" or self.method == "ldp":
            return self.method.upper()
        else:
            return self.method.title()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", "-r", type=str, default="data",
                        help="Path to data directory, where all the datasets will be placed. Default is 'data'")
    parser.add_argument("--name", "-n", type=str, default="cora",
                        help="Name of the dataset. Supported names are: cora, citeseer, pubmed, photo, computers, cs, and physics")
    parser.add_argument("--model", '-m', type=str, default="gcn",
                        help="The type of GNN architecture. Supported architectures are: gcn, gat, and sage. Default is gcn")
    parser.add_argument("--aug", '-a', type=str, default="split",
                        help="The name of data augmentation technique. Valid options are: ppr, hk, katz, split, zscore, ldp, paste. Default is split.")
    parser.add_argument("--layers", "-l", nargs="+", default=[
                        512, 128], help="The number of units of each layer of the GNN. Default is [512, 128]")
    parser.add_argument("--init-parts", "-ip", type=int, default=1,
                        help="The number of initial partitions. Default is 1. Applicable for ClusterSelfGNN")
    parser.add_argument("--final-parts", "-fp", type=int, default=1,
                        help="The number of final partitions. Default is 1. Applicable for ClusterSelfGNN")
    parser.add_argument("--heads", '-hd', nargs="+", type=int, default=[
                        8, 1], help="The number of heads of each layer of a GAT architecture. Default is [8, 1]. Applicable for gat model.")
    parser.add_argument("--lr", '-lr', type=float, default=0.0001,
                        help="Learning rate. Default is 0.0001.")
    parser.add_argument("--dropout", "-do", type=float,
                        default=0.2, help="Dropout rate. Default is 0.2")
    parser.add_argument("--cache-step", '-cs', type=int, default=1000,
                        help="The step size to cache the model, that is, every cache_step the model is persisted. Default is 100.")
    parser.add_argument("--epochs", '-e', type=int,
                        default=100, help="The number of epochs")
    return parser.parse_args()


def decide_config(root, name):
    """
    Create a configuration to download datasets
    :param root: A path to a root directory where data will be stored
    :param name: The name of the dataset to be downloaded
    :return: A modified root dir, the name of the dataset class, and parameters associated to the class
    """
    if name == 'cora' or name == 'citeseer' or name == "pubmed":
        root = osp.join(root, "pyg", "planetoid")
        params = {"kwargs": {"root": root, "name": name},
                  "name": name, "class": Planetoid, "src": "pyg"}
    elif name == 'computers' or name == "photo":
        root = osp.join(root, "pyg", name)
        params = {"kwargs": {"root": root, "name": name},
                  "name": name, "class": Amazon, "src": "pyg"}
    elif name == 'cs' or name == 'physics':
        root = osp.join(root, "pyg", name)
        params = {"kwargs": {"root": root, "name": name},
                  "name": name, "class": Coauthor, "src": "pyg"}
    return params


def create_dirs(dirs):
    for dir_tree in dirs:
        sub_dirs = dir_tree.split("/")
        path = ""
        for sub_dir in sub_dirs:
            path = osp.join(path, sub_dir)
            os.makedirs(path, exist_ok=True)


def create_masks(data):
    if not hasattr(data, "val_mask"):
        labels = data.y.numpy()
        counter = Counter(labels)
        dev_size = int(labels.shape[0] * 0.1)
        test_size = int(labels.shape[0] * 0.2)

        perm = np.random.permutation(labels.shape[0])
        start = end = 0
        test_labels = []
        dev_labels = []
        for l, c in counter.items():
            frac = c / labels.shape[0]
            ts = int(frac * test_size)
            ds = int(frac * dev_size)
            end += ts
            t_lbl = perm[start:end]
            test_labels.append(t_lbl)
            start = end
            end += ds
            d_lbl = perm[start:end]
            dev_labels.append(d_lbl)
            start = end

        test_index, dev_index = np.concatenate(
            test_labels), np.concatenate(dev_labels)
        data_index = np.arange(labels.shape[0])
        test_mask = torch.tensor(
            np.in1d(data_index, test_index), dtype=torch.bool)
        dev_mask = torch.tensor(
            np.in1d(data_index, dev_index), dtype=torch.bool)
        train_mask = ~(dev_mask + test_mask)
        data.train_mask = train_mask
        data.val_mask = dev_mask
        data.test_mask = test_mask
    return data


def evaluate(features, labels, test_rate=0.4, seed=0):
    sf = ShuffleSplit(5, test_size=test_rate, random_state=seed)
    clf = OneVsRestClassifier(
        LogisticRegression(solver='liblinear'), n_jobs=-1)
    results = []
    features = StandardScaler().fit_transform(features)
    for train_index, test_index in sf.split(features, labels):
        train_x = features[train_index]
        train_y = labels[train_index]
        test_x = features[test_index]
        test_y = labels[test_index]
        clf.fit(train_x, train_y)
        pred = clf.predict(test_x)
        acc = accuracy_score(test_y, pred)
        results.append(acc)
    return np.mean(results), np.std(results)


def get_device_id(cuda_is_available):
    if not cuda_is_available:
        return "cpu"
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]).decode(
        'utf-8')
    gpu_stats = gpu_stats.strip().split('\n')
    stats = []
    for i in range(1, len(gpu_stats)):
        info = gpu_stats[i].split()
        used = int(info[0])
        free = int(info[2])
        stats.append([used, free])
    stats = np.array(stats)
    gpu_index = stats[:, 1].argmax()
    available_mem_on_gpu = stats[gpu_index][1] - stats[gpu_index][0]
    return gpu_index if available_mem_on_gpu > 5000 else -1
