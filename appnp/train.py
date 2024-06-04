import numpy as np
from tqdm import trange
import pandas as pd
import random
import torch
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from scipy import sparse
from scipy import linalg
import torch_geometric
import os
from torch_geometric.datasets import Planetoid
import torch.optim as optim
import scipy.io as sio
import networkx as nx
from copy import deepcopy
from torch_geometric.utils import to_networkx
import itertools
import random as rand
import torch
import time
import datetime
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.autograd as autograd
from torch.nn.modules.container import ModuleList, ModuleDict, Module
from torch.nn.parameter import Parameter
from torch import Tensor
from typing import Union, Optional, Iterable, Dict, Tuple
from contextlib import contextmanager
import math
import json
import torch
import networkx as nx
from scipy import sparse
from texttable import Texttable
from torch import Tensor
import argparse
from torch_sparse import spmm


def parameter_parser():
    """
    A method to parse up command line parameters. By default it trains on the Cora dataset.
    The default hyperparameters give a good quality representation without grid search.
    """
    parser = argparse.ArgumentParser(description="Run PPNP/APPNP.")

    parser.add_argument('-f')

    #parser.add_argument("--edge-path",
    #                    nargs="?",
    #                    default="./input/cora_edges.csv",
	  #              help="Edge list csv.")

    parser.add_argument("--norm",
                        nargs="?",
                        default=np.inf,
	                help="Normalization norm")
    parser.add_argument("--weight_decay",
                        nargs="?",
                        default=1e-3,
	                help="Normalization norm")


    parser.add_argument("--features-path",
                        nargs="?",
                        default="/content/drive/MyDrive/APPNP/data_semisupervised/cora-cit/features.pickle",
	                help="Features json.")

    parser.add_argument("--target-path",
                        nargs="?",
                        default="/content/drive/MyDrive/APPNP/data_semisupervised/cora-cit/node-labels-cora-cit.txt",
	                help="Target classes csv.")

    parser.add_argument("--model",
                        nargs="?",
                        default="APPNP",
	                help="Model type.")

    parser.add_argument("--epochs",
                        type=int,
                        default=2000,
	                help="Number of training epochs. Default is 2000.")

    parser.add_argument("--seed",
                        type=int,
                        default=3,
	                help="Random seed for train-test split. Default is 12.")

    parser.add_argument("--iterations",
                        type=int,
                        default=10,
	                help="Number of Approximate Personalized PageRank iterations. Default is 10.")

    parser.add_argument("--early-stopping-rounds",
                        type=int,
                        default=500,
	                help="Number of training rounds before early stopping. Default is 10.")

    parser.add_argument("--knownlabels",
                        type=int,
                        default=0.05,
	                help="Known labels. Default is 5%")

    parser.add_argument("--dropout",
                        type=float,
                        default=0.3,
	                help="Dropout parameter. Default is 0.5.")

    parser.add_argument("--alpha",
                        type=float,
                        default=0.1,
	                help="Page rank teleport parameter. Default is 0.1.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.01,
	                help="Learning rate. Default is 0.01.")

    parser.add_argument("--lambd",
                        type=float,
                        default=0.005,
	                help="Weight matrix regularization. Default is 0.005.")

    parser.add_argument("--layers",
                        nargs="+",
                        type=int,
                        help="Layer dimensions separated by space. E.g. 64 64.")

    parser.set_defaults(layers=[64, 64])

    return parser.parse_args()

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())



def hyper_to_graph(graph,n):
  """
  hypergraph -> graph
  """
  import itertools
  A = np.zeros((n,n))
  for _, row in graph.iterrows():
    try:
      edge = row.tolist()[0]
      edge = [int(i) for i in edge.split(",")]
    except:
      edge = row.tolist()

    for i in list(itertools.combinations(edge, 2)):
      A[i[0]-1,i[1]-1], A[i[1]-1,i[0]-1] = 1,1


  return A

def graph_reader(graph,n):
    """
    Function to read the graph from the path.
    :param path: Path to the edge list.
    :return graph: NetworkX object returned.
    """
    # Usare k-nn non funziona !!

    #edges = [list(a) for a in edges]
    #graph = nx.from_edgelist(edges)

    #neigh = NearestNeighbors(n_neighbors=20)
    #neigh.fit(graph)
    #A = neigh.kneighbors_graph(graph)

    A = hyper_to_graph(graph,n)
    graph = nx.from_numpy_array(A)
    graph.remove_edges_from(nx.selfloop_edges(graph))

    #connected_comp = [c for c in sorted(nx.connected_components(graph), key=len, reverse=True)]
    #graph =graph.subgraph(connected_comp[0])

    return graph

#def feature_reader(features):
#    """
#    Reading the feature matrix stored as JSON from the disk.
#    :param path: Path to the JSON file.
#    :return out_features: Dict with index and value tensor.
#    """
#    features = {int(k): [int(val) for val in v] for k, v in features.items()}
#    return features

def feature_reader(features):
    """
    take in input a fetures matrix and return a dict
    values, rows,colums
    """
    #features = {int(k): [int(val) for val in v] for k, v in features.items()}
    rows = np.array(np.where(features!=0)[0],dtype=np.int32).tolist()
    columns = np.array(np.where(features!=0)[1],dtype=np.int32).tolist()
    values = features[rows,columns].flatten().tolist()

    return [values,rows,columns,features.shape[1]]

def target_reader(path):
    """
    Reading the target vector from disk.
    :param path: Path to the target.
    :return target: Target vector.
    """

    if path.split(".")[-1]=="pickle":
      target = np.array(pd.read_pickle(path))
    else:
      target = pd.read_csv(path,sep=", ",header=None)[0].tolist()
      target = [i-1 for i in target]
      target = np.array(target)
    return target

def create_adjacency_matrix(graph):
    """
    Creating a sparse adjacency matrix.
    :param graph: NetworkX object.
    :return A: Adjacency matrix.
    """
    index_1 = [edge[0] for edge in graph.edges()] + [edge[1] for edge in graph.edges()]
    index_2 = [edge[1] for edge in graph.edges()] + [edge[0] for edge in graph.edges()]
    values = [1 for edge in index_1]
    node_count = max(max(index_1)+1, max(index_2)+1)
    A = sparse.coo_matrix((values, (index_1, index_2)), shape=(node_count, node_count), dtype=np.float32)
    return A

def normalize_adjacency_matrix(A, I):
    """
    Creating a normalized adjacency matrix with self loops.
    :param A: Sparse adjacency matrix.
    :param I: Identity matrix.
    :return A_tile_hat: Normalized adjacency matrix.
    """
    A_tilde = A + I
    degrees = A_tilde.sum(axis=0)[0].tolist()
    D = sparse.diags(degrees, [0])
    D = D.power(-0.5)
    A_tilde_hat = D.dot(A_tilde).dot(D)
    return A_tilde_hat

def create_propagator_matrix(graph, alpha, model):
    """
    Creating  apropagation matrix.
    :param graph: NetworkX graph.
    :param alpha: Teleport parameter.
    :param model: Type of model exact or approximate.
    :return propagator: Propagator matrix Dense torch matrix /
    dict with indices and values for sparse multiplication.
    """
    print(graph)
    A = create_adjacency_matrix(graph)
    I = sparse.eye(A.shape[0])
    A_tilde_hat = normalize_adjacency_matrix(A, I)
    if model == "exact":
        propagator = (I-(1-alpha)*A_tilde_hat).todense()
        propagator = alpha*torch.inverse(torch.FloatTensor(propagator))
    else:
        propagator = dict()
        A_tilde_hat = sparse.coo_matrix(A_tilde_hat)
        indices = np.concatenate([A_tilde_hat.row.reshape(-1, 1), A_tilde_hat.col.reshape(-1, 1)], axis=1).T
        propagator["indices"] = torch.LongTensor(indices)
        propagator["values"] = torch.FloatTensor(A_tilde_hat.data)
    return propagator

def uniform(size, tensor):
    """
    Uniform weight initialization.
    :param size: Size of the tensor.
    :param tensor: Tensor initialized.
    """
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)

class DenseFullyConnected(torch.nn.Module):
    """
    Abstract class for PageRank and Approximate PageRank networks.
    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    :param density: Feature matrix structure.
    """
    def __init__(self, in_channels, out_channels):
        super(DenseFullyConnected, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        """
        Defining the weight matrices.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(self.out_channels))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        uniform(self.out_channels, self.bias)

    def forward(self, features):
        """
        Doing a forward pass.
        :param features: Feature matrix.
        :return filtered_features: Convolved features.
        """
        filtered_features = torch.mm(features, self.weight_matrix)
        filtered_features = filtered_features + self.bias
        return filtered_features

class SparseFullyConnected(torch.nn.Module):
    """
    Abstract class for PageRank and Approximate PageRank networks.
    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    :param density: Feature matrix structure.
    """
    def __init__(self, in_channels, out_channels):
        super(SparseFullyConnected, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        """
        Defining the weight matrices.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(self.out_channels))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        uniform(self.out_channels, self.bias)

    def forward(self, feature_indices, feature_values):
        """
        Making a forward pass.
        :param feature_indices: Non zero value indices.
        :param feature_values: Matrix values.
        :return filtered_features: Output features.
        """
        number_of_nodes = torch.max(feature_indices[0]).item()+1
        number_of_features = torch.max(feature_indices[1]).item()+1
        filtered_features = spmm(index = feature_indices,
                                 value = feature_values,
                                 m = number_of_nodes,
                                 n = number_of_features,
                                 matrix = self.weight_matrix)
        filtered_features = filtered_features + self.bias
        return filtered_features

class APPNPModel(torch.nn.Module):
    """
    APPNP Model Class.
    :param args: Arguments object.
    :param number_of_labels: Number of target labels.
    :param number_of_features: Number of input features.
    :param graph: NetworkX graph.
    :param device: CPU or GPU.
    """
    def __init__(self, args, number_of_labels, number_of_features, graph, device):
        super(APPNPModel, self).__init__()
        self.args = args
        self.number_of_labels = number_of_labels
        self.number_of_features = number_of_features
        self.graph = graph
        self.device = device
        self.setup_layers()
        self.setup_propagator()

    def setup_layers(self):
        """
        Creating layers.
        """
        self.layer_1 = SparseFullyConnected(self.number_of_features, self.args.layers[0])
        self.layer_2 = DenseFullyConnected(self.args.layers[1], self.number_of_labels)

    def setup_propagator(self):
        """
        Defining propagation matrix (Personalized Pagrerank or adjacency).
        """
        self.propagator = create_propagator_matrix(self.graph, self.args.alpha, self.args.model)
        if self.args.model == "exact":
            self.propagator = self.propagator.to(self.device)
        else:
            self.edge_indices = self.propagator["indices"].to(self.device)
            self.edge_weights = self.propagator["values"].to(self.device)

    def forward(self, feature_indices, feature_values):
        """
        Making a forward propagation pass.
        :param feature_indices: Feature indices for feature matrix.
        :param feature_values: Values in the feature matrix.
        :return self.predictions: Predicted class label log softmaxes.
        """
        feature_values = torch.nn.functional.dropout(feature_values,
                                                     p=self.args.dropout,
                                                     training=self.training)

        latent_features_1 = self.layer_1(feature_indices, feature_values)

        latent_features_1 = torch.nn.functional.relu(latent_features_1)

        latent_features_1 = torch.nn.functional.dropout(latent_features_1,
                                                        p=self.args.dropout,
                                                        training=self.training)

        latent_features_2 = self.layer_2(latent_features_1)
        if self.args.model == "exact":
            self.predictions = torch.nn.functional.dropout(self.propagator,
                                                           p=self.args.dropout,
                                                           training=self.training)

            self.predictions = torch.mm(self.predictions, latent_features_2)

        elif self.args.model == "APPNP tanh":
            localized_predictions = latent_features_2
            edge_weights = torch.nn.functional.dropout(self.edge_weights,
                                                       p=self.args.dropout,
                                                       training=self.training)

            for iteration in range(self.args.iterations):

                new_features = spmm(index=self.edge_indices,
                                    value=edge_weights,
                                    n=localized_predictions.shape[0],
                                    m=localized_predictions.shape[0],
                                    matrix=localized_predictions)

                localized_predictions = (1-self.args.alpha)*new_features
                localized_predictions = F.tanh(localized_predictions + self.args.alpha*latent_features_2)
            self.predictions = localized_predictions
        elif self.args.model == "APPNP normalized tanh":
            localized_predictions = latent_features_2
            edge_weights = torch.nn.functional.dropout(self.edge_weights,
                                                       p=self.args.dropout,
                                                       training=self.training)

            for iteration in range(self.args.iterations):

                new_features = spmm(index=self.edge_indices,
                                    value=edge_weights,
                                    n=localized_predictions.shape[0],
                                    m=localized_predictions.shape[0],
                                    matrix=localized_predictions)

                localized_predictions = (1-self.args.alpha)*new_features
                localized_predictions = F.tanh(localized_predictions + self.args.alpha*latent_features_2)
                localized_predictions = F.normalize(localized_predictions+1.2,dim=1,p=self.args.norm)

            self.predictions = localized_predictions
        self.predictions = torch.nn.functional.log_softmax(self.predictions, dim=1)
        return self.predictions
    
class APPNPTrainer(object):
    """
    Method to train PPNP/APPNP model.
    """
    def __init__(self, args,graph, features, target,number_of_features):
        """
        :param args: Arguments object.
        :param graph: Networkx graph.
        :param features: Feature matrix.
        :param target: Target vector with labels.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.graph = graph
        self.values = features[0]
        self.rows = features[1]
        self.columns = features[2]
        self.target = target
        self.number_of_features = number_of_features
        self.create_model()
        self.train_test_split()
        self.transfer_node_sets()
        self.process_features()
        self.transfer_features()

    def create_model(self):
        """
        Defining a model and transfering it to GPU/CPU.
        """
        self.node_count = self.graph.number_of_nodes()
        self.number_of_labels = np.max(self.target)+1
        #self.number_of_features = max([f for _, feats  in self.features.items() for f in feats])+1
        self.model = APPNPModel(self.args,
                                self.number_of_labels,
                                self.number_of_features,
                                self.graph,
                                self.device)

        self.model = self.model.to(self.device)

    def train_test_split(self):
        """
        Creating a train/test split.
        """
        #random.seed(self.args.seed)
        nodes = [node for node in range(self.graph.number_of_nodes())]
        #nodes = [node for node in range(self.node_count)]
        self.train_size = int(np.ceil(len(nodes)*self.args.knownlabels))
        self.test_size = int((len(nodes)-self.train_size)/2)
        random.shuffle(nodes)
        self.train_nodes = nodes[0:self.train_size]
        self.test_nodes = nodes[self.train_size:self.train_size+self.test_size]
        self.validation_nodes = nodes[self.train_size+self.test_size:]
        #random.seed(self.args.seed)
        #nodes = [node for node in range(self.graph.number_of_nodes())]
        ##nodes = [node for node in range(self.node_count)]
        #self.train_size = int(np.ceil(len(nodes)*self.args.knownlabels))
        #self.test_size = int((len(nodes)-self.train_size))
        #random.shuffle(nodes)
        #self.train_nodes = nodes[0:self.train_size]
        #self.test_nodes = nodes[self.train_size:]
        #self.validation_nodes = self.test_nodes



    def transfer_node_sets(self):
        """
        Transfering the node sets to the device.
        """
        self.train_nodes = torch.LongTensor(self.train_nodes).to(self.device)
        self.test_nodes = torch.LongTensor(self.test_nodes).to(self.device)
        self.validation_nodes = torch.LongTensor(self.validation_nodes).to(self.device)

    def process_features(self):
        """
        Creating a sparse feature matrix and a vector for the target labels.
        """
        #index_1 = [node for node in self.graph.nodes() for fet in self.features[node]]
        #index_2 = [fet for node in self.graph.nodes() for fet in self.features[node]]
        #values = [1.0/len(self.features[node]) for node in self.graph.nodes() for fet in self.features[node]]
        index_1 = self.rows
        index_2 = self.columns
        self.feature_indices = torch.LongTensor([index_1, index_2])
        self.feature_values = torch.FloatTensor(self.values)
        self.target = torch.LongTensor(self.target)

    def transfer_features(self):
        """
        Transfering the features and the target matrix to the device.
        """
        self.target = self.target.to(self.device)
        self.feature_indices = self.feature_indices.to(self.device)
        self.feature_values = self.feature_values.to(self.device)

    def score(self, index_set):
        """
        Calculating the accuracy for a given node set.
        :param index_set: Index of nodes to be included in calculation.
        :parm acc: Accuracy score.
        """
        self.model.eval()
        _, pred = self.model(self.feature_indices, self.feature_values).max(dim=1)
        correct = pred[index_set].eq(self.target[index_set]).sum().item()
        acc = correct / index_set.size()[0]
        return acc

    def do_a_step(self):
        """
        Doing an optimization step.
        """
        self.model.train()
        self.optimizer.zero_grad()
        prediction = self.model(self.feature_indices, self.feature_values)
        loss = torch.nn.functional.nll_loss(prediction[self.train_nodes],
                                            self.target[self.train_nodes])
        loss = loss+(self.args.lambd/2)*(torch.sum(self.model.layer_2.weight_matrix**2))
        loss.backward()
        self.optimizer.step()

    def train_neural_network(self):
        """
        Training a neural network.
        """
        print("\nTraining.\n")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate,weight_decay=self.args.weight_decay)
        self.best_accuracy = 0
        self.step_counter = 0
        iterator = trange(self.args.epochs, desc='Validation accuracy: ', leave=True)
        for _ in iterator:
            self.do_a_step()
            accuracy = self.score(self.validation_nodes)
            iterator.set_description("Validation accuracy: {:.4f}".format(accuracy))
            if accuracy >= self.best_accuracy:
                self.best_accuracy = accuracy
                self.test_accuracy = self.score(self.test_nodes)
                self.step_counter = 0
            else:
                self.step_counter = self.step_counter + 1
                if self.step_counter > self.args.early_stopping_rounds:
                    iterator.close()
                    break

    def fit(self):
        """
        Fitting the network and calculating the test accuracy.
        """
        self.train_neural_network()
        print("\nBreaking from training process because of early stopping.\n")
        print("Test accuracy: {:.4f}".format(self.test_accuracy))
        return self.test_accuracy