import numpy as np
import torch
from torch_geometric.data import Data
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import random

class MultiChannelsGCN(torch.nn.Module):
    
    def __init__(self, channels, input_dim, post_conv_dim, output_dim, identity=False):
        super(MultiChannelsGCN, self).__init__()
        self.identity = identity
        self.channels = channels
        self.input_dim = input_dim
        self.post_conv_dim = post_conv_dim
        self.output_dim = output_dim
        self.GCN = nn.ModuleList([GCNConv(input_dim, post_conv_dim) for _ in range(channels)])
        if identity:
            self.dense = nn.Linear(post_conv_dim * (channels + 1), output_dim)
            self.denseID = nn.Linear(input_dim, post_conv_dim)
        else:
            self.dense = nn.Linear(post_conv_dim * channels, output_dim)

    def forward(self, nodes, edges):
        X = []
        for k in range(self.channels):
            if len(edges[k]) == 0:
                x = torch.zeros(nodes.shape[0], self.post_conv_dim)
            else:
                x = F.relu(self.GCN[k](nodes, edges[k]))
            X.append(x)
        if self.identity:
            X.append(F.relu(self.denseID(nodes)))
        concat = torch.cat(X, dim=1)
        return F.relu(self.dense(concat))
    
class NodeClassifier(torch.nn.Module):
    def __init__(self, channels, input_dim):
        super(NodeClassifier, self).__init__()
        self.GCN1 = MultiChannelsGCN(channels, input_dim, 50, 20, identity=True)
        self.dense1 = nn.Linear(20,1)

    def forward(self, data):
        nodes, edges = data.x, data.edge_index
        x = self.GCN1(nodes, edges)
        x = self.dense1(x)
        x = torch.sigmoid(x)
        return x

    def predict(self, graph):
        self.eval()
        with torch.no_grad():
            logits = self.forward(graph)
        return np.array((logits > 0.5).int()).flatten()
    
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score

class ModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def fit(self, train_graph_dict, validation_graph_dict, verbose=1, max_epochs=10):
        # Training logic
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        # Use DataLoader to create batches of data
        train_loader = DataLoader(list(train_graph_dict.values()), batch_size=1, shuffle=True)
        N_train = len(train_loader)
        validation_loader = DataLoader(list(validation_graph_dict.values()), batch_size=1, shuffle=False)
        N_validation = len(validation_loader)

        if verbose > 0:
            print('Training on', N_train, 'graphs, validating on', N_validation, 'graphs')

        # Train the model
        model_name = "model_py_torch"
        best_f1_score = 0
        for epoch in range(max_epochs):
            if verbose > 0:
                print('- Epoch', f'{epoch + 1:03d}', '-')
            # training
            self.model.train()
            total_loss = 0
            for data in train_loader:
                data = data.to(device)
                self.optimizer.zero_grad()
                output = self.model(data).squeeze()
                loss = self.criterion(output, data.y.float())
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            average_loss = total_loss / N_train
            if verbose > 1:
                print('Loss:', f'{average_loss:.4f}')
            

            # Evaluate the model on the training set
            self.model.eval()
            f1_moyen_train = 0
            for data in train_loader:
                data = data.to(device)
                y_pred = self.model.predict(data)
                y_true = data.y.cpu().numpy()
                f1 = f1_score(y_true, y_pred)
                f1_moyen_train += f1
            f1_moyen_train /= N_train
            if verbose > 1:
                print('F1 train:', f1_moyen_train)

            # Evaluate the model on the validation set
            self.model.eval()
            f1_moyen_valid = 0
            for data in validation_loader:
                data = data.to(device)
                y_pred = self.model.predict(data)
                y_true = data.y.cpu().numpy()
                f1 = f1_score(y_true, y_pred)
                f1_moyen_valid += f1
            f1_moyen_valid /= N_validation
            if verbose > 1:
                print('F1 valid:', f1_moyen_valid)

            # callbacks ou autre
            if f1_moyen_valid > best_f1_score:
                if verbose > 1:
                    print('It\'s better !' )
                torch.save(self.model.state_dict(), "training_states/" + model_name + "-best.pth")
            else:
                self.optimizer.param_groups[0]['lr'] /= 2
                if verbose > 1:
                    print('Learning rate reduced to:', self.optimizer.param_groups[0]['lr'])
            if verbose > 1:
                print('')
        
        if verbose > 0:
            print('Training finished !')

        self.model.load_state_dict(torch.load("training_states/" + model_name + "-best.pth"))

    def predict(self, graphs_dict):
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        result = {}
        for key, graph in graphs_dict.items():
            data = graph.to(device)
            y_pred = self.model.predict(data)
            result[key] = y_pred
        return result


    def score(self, graphs_dict):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        validation_loader = DataLoader(list(graphs_dict.values()), batch_size=1, shuffle=False)
        N_validation = len(validation_loader)
        self.model.eval()
        f1_moyen_valid = 0
        for data in validation_loader:
            data = data.to(device)
            y_pred = self.model.predict(data)
            y_true = data.y.cpu().numpy()
            f1 = f1_score(y_true, y_pred)
            f1_moyen_valid += f1
        f1_moyen_valid /= N_validation
        return f1_moyen_valid


class BaggingModel(BaseEstimator, ClassifierMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, train_graph_dict, epochs, verbose=1):
        i = 0
        for model in self.models:
            i += 1
            if verbose > 0:
                print('Model', i)
            keys = list(train_graph_dict.keys())
            bagging_train_graphs = {}
            bagging_validation_graphs = {}
            samples = random.choices(keys, k=len(keys))
            c_train = 0
            c_validation = 0
            for key in keys:
                if key in samples:
                    bagging_train_graphs[c_train] = train_graph_dict[key]
                    c_train += 1
                else:
                    bagging_validation_graphs[c_validation] = train_graph_dict[key]
                    c_validation += 1
            model.fit(bagging_train_graphs, bagging_validation_graphs, max_epochs=epochs, verbose=0)
            if verbose > 0:
                print('F1 score (out of the bag):', model.score(bagging_validation_graphs))
        
    def predict(self, graphs_dict):
        result = {}
        for key, graph in graphs_dict.items():
            y_pred = 0
            for model in self.models:
                y_pred += model.predict({key: graph})[key]
            y_pred =  y_pred / len(self.models)
            y_pred = np.array((y_pred > 0.5).astype(int)).flatten()
            result[key] = y_pred
        return result
        


    def score(self, validation_graphs_dict):
        f1_moyen_valid = 0
        for key, graph in validation_graphs_dict.items():
            y_pred = 0
            for model in self.models:
                y_pred += model.predict({key: graph})[key]
            y_pred =  y_pred / len(self.models)
            y_pred = np.array((y_pred > 0.5).astype(int)).flatten()
            y_true = graph.y.cpu().numpy()
            f1 = f1_score(y_true, y_pred)
            f1_moyen_valid += f1
        f1_moyen_valid /= len(validation_graphs_dict)
        return f1_moyen_valid
    