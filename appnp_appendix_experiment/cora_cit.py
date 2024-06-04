import train
import numpy as np
import pandas as pd
import torch
import random

parser = train.parameter_parser()


parser.dropout = 0.50
parser.learning_rate = 0.01
parser.alpha = 0.1
parser.model = "APPNP tanh"
parser.weight_decay = 0

alpha_list = [0.05,0.1,0.3,0.5,0.7,0.9]

path = "data_graph"
parser.target_path = path+"/cora-cit/node-labels-cora-cit.txt"
x = pd.read_pickle(path+"/cora-cit/features.pickle").toarray()
hyper = pd.read_csv(path+"/cora-cit/hyperedges-cora-cit.txt",sep=", ", header=None)
n = x.shape[0]
parser.knownlabels = 0.078

print('=================================================================================')
print(f'APPNP (Tanh) 2')
print('=================================================================================')
print("\n")



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
    trainer = train.APPNPTrainer(args, graph, features, target,test_nodes,number_of_features = features[3])
    return trainer.fit()

torch.manual_seed(3)
torch.cuda.manual_seed(3)
np.random.seed(3)
random.seed(3)


graph = train.graph_reader(hyper,n)
test_nodes = train.test_split(0.052,graph)

val_acc =[]

for i in alpha_list:

  parser.alpha = i
  val_acc.append(main()[0])

parser.alpha = alpha_list[np.argmax(val_acc)]
print(alpha_list[np.argmax(val_acc)])

val = []
for _ in range(5):

  val.append(main()[1])


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
print(f'APPNP(Normalized Tanh) 2')
print('=================================================================================')
print("\n")

parser.norm = np.inf

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
    trainer = train.APPNPTrainer(args, graph, features, target,test_nodes,number_of_features = features[3])
    return trainer.fit()

torch.manual_seed(3)
torch.cuda.manual_seed(3)
np.random.seed(3)
random.seed(3)


graph = train.graph_reader(hyper,n)
test_nodes = train.test_split(0.052,graph)

val_acc =[]

for i in alpha_list:

  parser.alpha = i
  val_acc.append(main()[0])

parser.alpha = alpha_list[np.argmax(val_acc)]
print(alpha_list[np.argmax(val_acc)])

val = []
for _ in range(5):

  val.append(main()[1])



mean = np.mean(val)
std = np.std(val)

mean = np.mean(val)
std = np.std(val)

print('=================================================================================')
print(f'Training (Results)')
print('=================================================================================')
print(f"Mean Accuracy: {mean}")
print(f"Standard deviation Accuracy: {std}")
print('=================================================================================')
print("\n")
