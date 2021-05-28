# SelfGNN

A PyTorch implementation of "SelfGNN: Self-supervised Graph Neural Networks without explicit negative sampling" [paper](https://arxiv.org/abs/2103.14958), which will appear in The International Workshop on Self-Supervised Learning for the Web (SSL'21) @ the Web Conference 2021 (WWW'21).

## Note
This is an ongoing work and the repository is subjected to continuous updates.



## Requirements!

-   Python 3.6+
-   PyTorch 1.6+
-   PyTorch Geometric 1.6+
-   Numpy 1.17.2+
-   Networkx 2.3+
-   SciPy 1.5.4+

## Example usage

```sh
$ python src/train.py
```

## :collision: Updates 

## update 2:

<style type="text/css">
.tg  {border-collapse:collapse;border-color:#9ABAD9;border-spacing:0;}
.tg td{background-color:#EBF5FF;border-color:#9ABAD9;border-style:solid;border-width:1px;color:#444;
  font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{background-color:#409cff;border-color:#9ABAD9;border-style:solid;border-width:1px;color:#fff;
  font-family:Arial, sans-serif;font-size:14px;font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-svo0{background-color:#D2E4FC;border-color:inherit;text-align:center;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow">Modules</th>
    <th class="tg-c3ow" colspan="15">Normalization Strategy</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-svo0">Encoder</td>
    <td class="tg-svo0" colspan="4">BN</td>
    <td class="tg-svo0" colspan="4">LN</td>
    <td class="tg-svo0" colspan="4">-</td>
    <td class="tg-svo0" colspan="3">-</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Projector</td>
    <td class="tg-c3ow" colspan="2">BN</td>
    <td class="tg-c3ow" colspan="2">-</td>
    <td class="tg-c3ow" colspan="2">LN</td>
    <td class="tg-c3ow" colspan="2">-</td>
    <td class="tg-c3ow" colspan="2">BN</td>
    <td class="tg-c3ow" colspan="2">LN</td>
    <td class="tg-c3ow" colspan="3">-</td>
  </tr>
  <tr>
    <td class="tg-svo0">Predictor</td>
    <td class="tg-svo0">BN</td>
    <td class="tg-svo0">-</td>
    <td class="tg-svo0">BN</td>
    <td class="tg-svo0">-</td>
    <td class="tg-svo0">LN</td>
    <td class="tg-svo0">-</td>
    <td class="tg-svo0">LN</td>
    <td class="tg-svo0">-</td>
    <td class="tg-svo0">BN</td>
    <td class="tg-svo0">-</td>
    <td class="tg-svo0">LN</td>
    <td class="tg-svo0">-</td>
    <td class="tg-svo0">BN</td>
    <td class="tg-svo0">LN</td>
    <td class="tg-svo0">-</td>
  </tr>
  <tr>
    <td class="tg-c3ow">SelfGNN</td>
    <td class="tg-c3ow">94.05±0.23</td>
    <td class="tg-c3ow">94.2±0.17</td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">94.01</span>±0.20</td>
    <td class="tg-c3ow">93.9±0.18</td>
    <td class="tg-c3ow">81.42±2.43</td>
    <td class="tg-c3ow">84.1±1.58</td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">92.39</span>±0.38</td>
    <td class="tg-c3ow">91.93±0.40</td>
    <td class="tg-c3ow">90.01±0.09</td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">90.12</span>±0.07</td>
    <td class="tg-c3ow">45.34±2.47</td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">52.92</span>±3.37</td>
    <td class="tg-c3ow">91.13±0.13</td>
    <td class="tg-c3ow">50.64±2.84</td>
    <td class="tg-c3ow">50.35±2.73</td>
  </tr>
</tbody>
</table>

## update 1:

- Both the paper and the source code are updated following the discussion on [this issue](https://github.com/zekarias-tilahun/SelfGNN/issues/1)
- Ablation study on the impact of ```BatchNorm``` added following reviewers feedback from SSL'21
   - The findings show that ```SelfGNN``` with out batch normalization is not stable and often its performance drops significantly
   - Layer Normalization behaves similar to the finding of no ```BatchNorm```

   
## Possible options for training SelfGNN


The following options can be passed to `src/train.py`

`--root:` or `-r:`
A path to a root directory to put all the datasets. Default is `./data`

`--name:` or `-n:`
The name of the datasets. Default is `cora`. Check the [`Supported dataset names`](#Supported-dataset-names)

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

`--cache-step:` or `-cs:`
The step size for caching the model. That is, every `--cache-step` the model will be persisted. Default is 100.

`--init-parts:` or `-ip:`
The number of initial partitions, for using the improved version using Clustering.
Default is 1.

`--final-parts:` or `-fp:`
The number of final partitions, for using the improved version using Clustering.
Default is 1.

## Supported dataset names

| name        | Description | Edges   | Features | Classes | Description                   |
| ----------- | ----------- | ------- | -------- | ------- | ----------------------------- |
| `Cora`      | 2,708       | 5,278   | 1,433    | 7       | Citation Network              |
| `Citeseer`  | 3,327       | 4,552   | 3,703    | 6       | Citation Network              |
| `Pubmed`    | 19,717      | 44,324  | 500      | 3       | Citation Network              |
| `Photo`     | 7,487       | 119,043 | 745      | 8       | Co-purchased products network |
| `Computers` | 13,381      | 245,778 | 767      | 10      | Co-purchased products network |
| `CS`        | 18,333      | 81,894  | 6,805    | 15      | Collaboration network         |
| `Physics`   | 34,493      | 247,962 | 8,415    | 5       | Collaboration network         |


Any dataset from the PyTorch Geometric library can be used, however `SelfGNN` is tested only on
the above datasets.


Citing
------

If you find this research helpful, please cite it as

```
@misc{kefato2021selfsupervised,
      title={Self-supervised Graph Neural Networks without explicit negative sampling}, 
      author={Zekarias T. Kefato and Sarunas Girdzijauskas},
      year={2021},
      eprint={2103.14958},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```