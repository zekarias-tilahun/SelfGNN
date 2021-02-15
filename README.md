# SelfGNN
A PyTorch implementation of SelfGNN: Self-supervised Graph Neural Networks without explicit negative sampling

Requirements!
-------------
  - Python 3.6+
  - PyTorch 1.6+
  - PyTorch Geometric 1.6+
  - Numpy 1.17.2+
  - Networkx 2.3+
  - SciPy 1.5.4+ 

Example usage
-------------

```sh
$ python src/train.py
```

The following options can be specified for src/train.py

`--root:` or `-r:` 
A path to a root directory to put all the datasets. Default is ```./data```

`--name:` or `-n:`
The name of the datasets. Default is ```cora```. Check the [```Supported dataset names```](#Supported-dataset-names) 

`--model:` or `-m:`
The type of GNN architecture to use. Curently three architectres are supported (gcn, gat, sage). 
Default is gcn.

`--aug:` or `-a:`
The name of the data augmentation technique. Curently (ppr, heat, katz, split, zscore, ldp, paste) are supported.
Default is split.

`--layers:` or `-l:`
One or more integer values specifying the number of units for each GNN layer.
Default is 512 128

`--heads:` or `-hd:`
One or more values specifying the number of heads for each GAT layer.
Applicable for `--model gat`. Default is 8 1

`--lr:` or `-lr:`
Learning rate, a value in [0, 1]. Default is 0.0001

`--dropout:` or `-do:`
Dropout rate, a value in [0, 1]. Deafult is 0.2

`--epochs:` or `-e:`
The number of epochs. Default is 1000.

`--init-parts:` or `-ip:`
The number of initial partitions, for using the improved version using Clustering.
Default is 1.

`--final-parts:` or `-fp:`
The number of final partitions, for using the improved version using Clustering.
Default is 1.

Supported dataset names
-----------------------
 - ```cora``` (Citation dataset)
 - ```citeseer``` (Citation dataset)
 - ```pubmed``` (Citation dataset)
 - ```computers``` (Co-purchased products from Amazon computers category)
 - ```photo``` (Co-purchased products from Amazon computers category)
 - ```physics``` (Co-authorship graph from the physics category based on the Microsoft Academic Graph from the KDD Cup 2016 challenge)
 - ```cs``` (Co-authorship graph from the computer science category based on the Microsoft Academic Graph from the KDD Cup 2016 challenge)
