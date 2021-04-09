from torch_geometric.data import DataLoader

import numpy as np

import torch

import models
import utils
import data

import os.path as osp
import os
import sys


class ModelTrainer:

    def __init__(self, args):
        self._args = args
        self._init()

    def _init(self):
        args = self._args
        self._device = torch.device(
            utils.get_device_id(torch.cuda.is_available()))
        self._aug = utils.Augmentations(method=args.aug)

        self._dataset = data.Dataset(root=args.root, name=args.name, num_parts=args.init_parts,
                                     final_parts=args.final_parts, augumentation=self._aug)
        self._loader = DataLoader(
            dataset=self._dataset)  # [self._dataset.data]
        print(f"Data Augmentation method {args.aug}")
        print(f"Data: {self._dataset.data}")
        hidden_layers = [int(l) for l in args.layers]
        layers = [self._dataset.data.x.shape[1]] + hidden_layers
        self._model = models.SelfGNN(layer_config=layers, dropout=args.dropout, gnn_type=args.model,
                                     heads=args.heads).to(self._device)
        print(self._model)
        self._optimizer = torch.optim.Adam(
            params=self._model.parameters(), lr=args.lr)

    def train(self):
        self._model.train()
        for epoch in range(self._args.epochs):
            for bc, batch_data in enumerate(self._loader):
                batch_data.to(self._device)
                v1_output, v2_output, loss = self._model(
                    x1=batch_data.x, x2=batch_data.x2, edge_index_v1=batch_data.edge_index, edge_index_v2=batch_data.edge_index2,
                    edge_weight_v1=batch_data.edge_attr, edge_weight_v2=batch_data.edge_attr2)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                self._model.update_moving_average()
                sys.stdout.write('\rEpoch {}/{}, batch {}/{}, loss {:.4f}'.format(epoch + 1, self._args.epochs, bc + 1,
                                                                                  self._dataset.final_parts, loss.data))
                sys.stdout.flush()
            if (epoch + 1) % self._args.cache_step == 0:
                path = osp.join(self._dataset.model_dir,
                                f"model.ep.{epoch + 1}.pt")
                torch.save(self._model.state_dict(), path)
        print()

    def infer(self):
        outputs = []
        self._model.train(False)
        for bc, batch_data in enumerate(self._loader):
            batch_data.to(self._device)
            v1_output, v2_output, _ = self._model(
                x1=batch_data.x, x2=batch_data.x2,
                edge_index_v1=batch_data.edge_index,
                edge_index_v2=batch_data.edge_index2,
                edge_weight_v1=batch_data.edge_attr,
                edge_weight_v2=batch_data.edge_attr2)
            batch_out = torch.cat([v1_output, v2_output],
                                  dim=1).detach().cpu().numpy()
            batch_nodes = batch_data.nodes.cpu().numpy()
            batch_y = batch_data.y.detach().cpu().numpy()
            batch_trm = batch_data.train_mask.detach().cpu().numpy()
            batch_dem = batch_data.dev_mask.detach().cpu().numpy()
            batch_tem = batch_data.test_mask.detach().cpu().numpy()
            output = list(zip(batch_nodes, batch_out, batch_y,
                              batch_trm, batch_dem, batch_tem))
            outputs += output

        outputs = sorted(outputs, key=lambda l: l[0])
        _, embeddings, y, train_mask, dev_mask, test_mask = zip(*outputs)

        self._embeddings = np.stack(embeddings)
        self._labels = np.array(y)
        self._train_mask = np.array(train_mask)
        self._dev_mask = np.array(dev_mask)
        self._test_mask = np.array(test_mask)
        
    def infer_embeddings(self):
        outputs = []
        self._model.train(False)
        self._embeddings = self._labels = None
        self._train_mask = self._dev_mask = self._test_mask = None
        for bc, batch_data in enumerate(self._loader):
            batch_data.to(self._device)
            v1_output, v2_output, _ = self._model(
                x1=batch_data.x, x2=batch_data.x2,
                edge_index_v1=batch_data.edge_index,
                edge_index_v2=batch_data.edge_index2,
                edge_weight_v1=batch_data.edge_attr,
                edge_weight_v2=batch_data.edge_attr2)
            emb = torch.cat([v1_output, v2_output], dim=1).detach()
            y = batch_data.y.detach()
            trm = batch_data.train_mask.detach()
            dem = batch_data.dev_mask.detach()
            tem = batch_data.test_mask.detach()
            if self._embeddings is None:
                self._embeddings, self._labels = emb, y
                self._train_mask, self._dev_mask, self._test_mask = trm, dem, tem
            else:
                self._embeddings = torch.cat([self._embeddings, emb])
                self._labels = torch.cat([self._labels, y])
                self._train_mask = torch.cat([self._train_mask, trm])
                self._dev_mask = torch.cat([self._dev_mask, dem])
                self._test_mask = torch.cat([self._test_mask, tem])

    def evaluate_epoch(self, epoch):
        path = osp.join(self._dataset.model_dir, f"model.ep.{epoch}.pt")
        self._model.load_state_dict(
            torch.load(path, map_location=self._device))
        self.infer_embeddings()
        dev_acu = self.evaluate_semi(partition="dev")
        test_acu = self.evaluate_semi(partition="test")
        return dev_acu, test_acu

    def search_best_epoch(self):
        model_files = os.listdir(self._dataset.model_dir)
        results = []
        best_epoch = -1, (0,), (0,)
        for i, model_file in enumerate(model_files):
            if model_file.endswith(".pt"):
                substr = model_file.split(".")
                epoch = int(substr[substr.index("ep") + 1])
                dev_acu, test_acu = self.evaluate_epoch(epoch)
                results.append([epoch, dev_acu, test_acu])
                if dev_acu[0] > best_epoch[1][0]:
                    best_epoch = epoch, dev_acu, test_acu
                print(epoch, dev_acu, test_acu)

        dev_accuracy, dev_std = best_epoch[1]
        test_accuracy, test_std = best_epoch[2]
        print(f"The best epoch is: {best_epoch[0]}")
        print(f"with validation accuracy: {dev_accuracy} std: {dev_std}")
        print(f"with test accuracy: {test_accuracy} std: {test_std}")
        path = osp.join(self._dataset.result_dir, "results-bn.txt")
        has_header = False
        algorithm = f"SelfGNN-{str(self._aug)}"
        with open(path, "a") as f:
            # Entry Dataset,Group,DevAccuracy,DevStd,TestAccuracy,TestStd,Algorithm
            f.write(
                f"{self._dataset.name.title()},{self._args.model.upper()},{dev_accuracy},{dev_std},{test_accuracy},{test_std},{algorithm}\n")

    def evaluate_semi(self, partition="dev"):
        """
        Used for producing the results of Experiment 1 in the paper. 
        """
        if partition == "train":
            mask = self._train_mask
        elif partition == "dev":
            mask = self._dev_mask
        else:
            mask = self._test_mask
        
        features = self._embeddings[mask].detach().cpu().numpy()
        labels = self._labels[mask].detach().cpu().numpy()
        dev_result, test_result = utils.evaluate(features, labels)
        return dev_result, test_result
    
    def evaluate(self):
        """
        Used for producing the results of Experiment 3 in the paper. 
        """
        print("Evaluating ...")
        emb_dim, num_class = self._embeddings.shape[1], self._labels.unique().shape[0]
        loss_fn = 
        dev_accs, test_accs = [], []
        
        for i in range(50):
            classifier = models.LogisticRegression(emb_dim, num_class).to(self._device)
            optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01, weight_decay=0.0)

            for _ in range(100):
                classifier.train()
                logits, loss = classifier(self._embeddings[self._train_mask], self._labels[self._train_mask])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            dev_logits, _ = classifier(self._embeddings[self._dev_mask], self._labels[self._dev_mask])
            test_logits, _ = classifier(self._embeddings[self._test_mask], self._labels[self._test_mask])
            dev_preds = torch.argmax(dev_logits, dim=1)
            test_preds = torch.argmax(test_logits, dim=1)
            
            dev_acc = (torch.sum(dev_preds == self._labels[self._dev_mask]).float() / self._labels[self._dev_mask].shape[0]).detach().cpu().numpy()
            test_acc = (torch.sum(test_preds == self._labels[self._test_mask]).float() / self._labels[self._test_mask].shape[0]).detach().cpu().numpy()
            dev_accs.append(dev_acc * 100)
            test_accs.append(test_acc * 100)
            print("Finished iteration {:02} of the logistic regression classifier. Validation accuracy {:.2f} test accuracy {:.2f}".format(i + 1, dev_acc, test_acc))

        dev_accs = np.stack(dev_accs)
        test_accs = np.stack(test_accs)
        
        print('Average validation accuracy: {:.2f} with std: {}'.format(dev_accs.mean(), dev_accs.std()))
        print('Average test accuracy: {:.2f} with std: {:.2f}'.format(test_accs.mean(), test_accs.std()))
        
        
def train_search_eval(args):
    trainer = ModelTrainer(args)
    trainer.train()
    trainer.search_best_epoch()
    

def train_eval(args):
    trainer = ModelTrainer(args)
    trainer.train()
    trainer.infer_embeddings()
    trainer.evaluate()


def main():
    args = utils.parse_args()
    train_eval(args)


if __name__ == "__main__":
    main()
