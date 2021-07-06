import logging
import optuna
import sys
import yaml

import torch

import train
import utils

import os.path as osp

        
def objective(trial):
    args.lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    args.dropout = trial.suggest_float("dropout", 0.0, 1.0)
    args.aug = trial.suggest_categorical("aug", ["ppr", "heat", "katz", "split", "zscore", "paste"])
    encoder_norm = trial.suggest_categorical("encoder_norm", ["no", "batch", "layer"])
    proj_norm = trial.suggest_categorical("proj_norm", ["no", "batch", "layer"])
    pred_norm = trial.suggest_categorical("pred_norm", ["no", "batch", "layer"])
    args.norms = [encoder_norm, proj_norm, pred_norm]
    hidden_layer = trial.suggest_int("hidden_layer", 32, 512, log=True)
    args.layers = [hidden_layer, 128]
    args.epochs = 50

    print(args)

    trainer = train.ModelTrainer(args)
    trainer.train()
    trainer.infer_embeddings()
    val_acc, _, _, _ = trainer.evaluate()

    return val_acc

    
args = utils.parse_args()
root = osp.expanduser(args.root)
params_path = osp.join(root, args.name, "processed", "tuned_params.yml")

optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=args.trials)


print("Best trial:")
trial = study.best_trial

print(f"  Value: {trial.value}")

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

with open(params_path, 'w') as f:
    yaml.safe_dump(trial.params, f)
