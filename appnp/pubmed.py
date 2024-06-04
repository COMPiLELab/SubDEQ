import train
import numpy as np
import pandas as pd
import torch
import random

parser = train.parameter_parser()

parser.dropout = 0.50
parser.learning_rate = 0.01
parser.alpha = 0.1
parser.norm = np.inf
parser.weight_decay = 0

path = "data_graph"
parser.target_path = path+"/pubmed/node-labels-pubmed.txt"
x = pd.read_pickle(path+"/pubmed/features.pickle").toarray()
hyper = pd.read_csv(path+"/pubmed/hyperedges-pubmed.txt",sep=", ", header=None)
n = x.shape[0]

parser.knownlabels = 0.008

print('=================================================================================')
print(f'APPNP (Tanh)')
print('=================================================================================')
print("\n")

parser.model = "APPNP tanh"

def main():
    """
    Parsing command line parameters, reading data, fitting an APPNP/PPNP and scoring the model.
    """
    args = parser
    torch.manual_seed(args.seed)
    train.tab_printer(args)
    graph = train.graph_reader(hyper,n)
    features = train.feature_reader(x)
    target = train.target_reader(args.target_path)
    trainer = train.APPNPTrainer(args, graph, features, target,number_of_features = features[3])
    return trainer.fit()

torch.manual_seed(3)
torch.cuda.manual_seed(3)
np.random.seed(3)
random.seed(3)

val = []
for _ in range(5):
  val.append(main())


mean = np.mean(val)
std = np.std(val)

print('=================================================================================')
print(f'Training (Results)')
print('=================================================================================')
print(f"Mean Accuracy: {mean}")
print(f"Standard deviation Accuracy: {std}")
print('=================================================================================')
print("\n")

print('=================================================================================')
print(f'APPNP (Normalized Tanh)')
print('=================================================================================')
print("\n")

parser.model = "APPNP normalized tanh"

def main():
    """
    Parsing command line parameters, reading data, fitting an APPNP/PPNP and scoring the model.
    """
    args = parser
    torch.manual_seed(args.seed)
    train.tab_printer(args)
    graph = train.graph_reader(hyper,n)
    features = train.feature_reader(x)
    target = train.target_reader(args.target_path)
    trainer = train.APPNPTrainer(args, graph, features, target,number_of_features = features[3])
    return trainer.fit()

torch.manual_seed(3)
torch.cuda.manual_seed(3)
np.random.seed(3)
random.seed(3)

val = []
for _ in range(5):
  val.append(main())


mean = np.mean(val)
std = np.std(val)


print('=================================================================================')
print(f'Training (Results)')
print('=================================================================================')
print(f"Mean Accuracy: {mean}")
print(f"Standard deviation Accuracy: {std}")
print('=================================================================================')
print("\n")