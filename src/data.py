from torch_geometric.datasets import Planetoid, Coauthor, Amazon, WikiCS, Actor
from torch_geometric.data import Data, ClusterData, InMemoryDataset
from torch_geometric.utils import subgraph

from tqdm import tqdm

import numpy as np
import torch.nn.functional as F
import torch

import os.path as osp
import sys

import utils


class Dataset(InMemoryDataset):

    """
    A PyTorch InMemoryDataset to build multi-view dataset through graph data augmentation
    """

    def __init__(self, root="data", name='cora', num_parts=1, final_parts=1, augumentation=None, transform=None,
                 pre_transform=None):
        self.num_parts = num_parts
        self.final_parts = final_parts
        self.augumentation = augumentation
        super().__init__(root=osp.join(root, name), transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    def download(self):
        utils.create_dirs(self.dirs)
        dataset = fetch_dataset(*osp.split(self.root))
        utils.create_masks(data=dataset.data)
        data = dataset.data
        edge_attr = torch.ones(data.edge_index.shape[1]) if data.edge_attr is None else data.edge_attr
        data.edge_attr = edge_attr
        torch.save((dataset.data, dataset.slices), self.processed_paths[1])

    def process(self):
        """
        Process either a full batch or cluster data.

        :return:
        """
        data, _ = torch.load(self.processed_paths[1])
        if self.num_parts == 1:
            data_list = self.process_full_batch_data(data)
        else:
            data_list = self.process_cluster_data(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
    def process_full_batch_data(self, view1data):
        """
        Augmented view data generation using the full-batch data.

        :param view1data:
        :return:
        """
        print("Processing full batch data")
        view2data = view1data if self.augumentation is None else self.augumentation(view1data)
        diff = abs(view2data.x.shape[1] - view1data.x.shape[1])
        if diff > 0:
            """
            Data augmentation on the features could lead to mismatch between the shape of the two views,
            hence the smaller view should be padded with zero. (smaller_data is a reference, changes will
            reflect on the original data)
            """
            smaller_data = view1data if view1data.x.shape[1] < view2data.x.shape[1] else view2data
            smaller_data.x = F.pad(smaller_data.x, pad=(0, diff))
            view1data.x = F.normalize(view1data.x)
            view2data.x = F.normalize(view2data.x)
        
        nodes = torch.tensor(np.arange(view1data.num_nodes), dtype=torch.long)
        data = Data(nodes=nodes, edge_index=view1data.edge_index, edge_index2=view2data.edge_index,
                    edge_attr=view1data.edge_attr,
                    edge_attr2=view2data.edge_attr, x=view1data.x, x2=view2data.x, y=view1data.y,
                    train_mask=view1data.train_mask,
                    dev_mask=view1data.val_mask, test_mask=view1data.test_mask, num_nodes=view1data.num_nodes)
        return [data]

    def process_cluster_data(self, data):
        """
        Data processing for ClusterSelfGNN. First the data object will be clustered according to the number of partition
        specified by this class. Then, we randomly sample a number of clusters and merge them together. Finally, data 
        augmentation is applied each of the final clusters. This is a simple strategy motivated by ClusterGCN and 
        employed to improve the scalability of SelfGNN.

        :param data: A PyTorch Geometric Data object
        :return: a list of Data objects depending on the final number of clusters.
        """
        data_list = []
        clusters = []
        num_parts, cluster_size = self.num_parts, self.num_parts // self.final_parts

        # Cluster the data
        cd = ClusterData(data, num_parts=num_parts)
        for i in range(1, cd.partptr.shape[0]):
            cls_nodes = cd.perm[cd.partptr[i - 1]: cd.partptr[i]]
            clusters.append(cls_nodes)

        # Randomly merge clusters and apply transformation
        np.random.shuffle(clusters)
        for i in tqdm(range(0, len(clusters), cluster_size), "Processing clusters"):
            end = i + cluster_size if len(clusters) - i > cluster_size else len(clusters)
            cls_nodes = torch.cat(clusters[i:end]).unique()

            x = data.x[cls_nodes]
            y = data.y[cls_nodes]
            train_mask = data.train_mask[cls_nodes]
            dev_mask = data.val_mask[cls_nodes]
            test_mask = data.test_mask[cls_nodes]
            edge_index, edge_attr = subgraph(cls_nodes, data.edge_index, relabel_nodes=True)
            view1data = Data(edge_index=edge_index, x=x, edge_attr=edge_attr, num_nodes=cls_nodes.shape[0])
            view2data = view1data if self.augumentation is None else self.augumentation(view1data)
            if not hasattr(view2data, "edge_attr") or view2data.edge_attr is None:
                view2data.edge_attr = torch.ones(view2data.edge_index.shape[1])
            diff = abs(view2data.x.shape[1] - view1data.x.shape[1])
            if diff > 0:
                smaller_data = view1data if view1data.x.shape[1] < view2data.x.shape[1] else view2data
                smaller_data.x = F.pad(smaller_data.x, pad=(0, diff))
                view1data.x = F.normalize(view1data.x)
                view2data.x = F.normalize(view2data.x)
            new_data = Data(y=y, x=view1data.x, x2=view2data.x, edge_index=view1data.edge_index,
                            edge_index2=view2data.edge_index,
                            edge_attr=view1data.edge_attr, edge_attr2=view2data.edge_attr, train_mask=train_mask,
                            dev_mask=dev_mask, test_mask=test_mask, num_nodes=cls_nodes.shape[0], nodes=cls_nodes)
            data_list.append(new_data)
        print()
        return data_list
    
    @property
    def name(self):
        return osp.split(self.root)[1]

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        if self.num_parts == 1:
            return [f'byg.data.aug.{self.augumentation.method}.pt', "data.pt"]
        else:
            return [f'byg.data.aug.{self.augumentation.method}.ip.{self.num_parts}.fp.{self.final_parts}.pt', "data.pt"]

    @property
    def model_dir(self):
        return osp.join(self.root, "model")

    @property
    def result_dir(self):
        return osp.join(self.root, "result")

    @property
    def dirs(self):
        return [self.raw_dir, self.processed_dir, self.model_dir, self.result_dir]

    
def fetch_dataset(root, name):
    """
    Fetchs datasets from the PyTorch Geometric library
    
    :param root: A path to the root directory a dataset will be placed
    :param name: Name of the dataset. Currently, the following names are supported
                'cora', 'citeseer', "pubmed", 'Computers', "Photo", 'CS',  'Physics'
    :return: A PyTorch Geometric dataset
    """
    print(name.lower())
    if name.lower() in {'cora', 'citeseer', "pubmed"}:
        return Planetoid(root=root, name=name)
    elif name.lower() in {'computers', "photo"}:
        return Amazon(root=root, name=name)
    elif name.lower() in {'cs',  'physics'}:
        return Coauthor(root=root, name=name)
    elif name.lower() == "wiki":
        return WikiCS(osp.join(root, "WikiCS"))
    elif name.lower() == "actor":
        return Actor(osp.join(root, name))
