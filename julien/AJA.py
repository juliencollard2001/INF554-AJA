from sklearn.metrics import confusion_matrix
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm 
from torch_geometric.data import Data
import random
from torch_geometric.loader import DataLoader

def flatten(list_of_list):
    return [item for sublist in list_of_list for item in sublist]

#####
# training and test sets of transcription ids
#####
training_set = ['ES2002', 'ES2005', 'ES2006', 'ES2007', 'ES2008', 'ES2009', 'ES2010', 'ES2012', 'ES2013', 'ES2015', 'ES2016', 'IS1000', 'IS1001', 'IS1002', 'IS1003', 'IS1004', 'IS1005', 'IS1006', 'IS1007', 'TS3005', 'TS3008', 'TS3009', 'TS3010', 'TS3011', 'TS3012']
training_set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in training_set])
training_set.remove('IS1002a')
training_set.remove('IS1005d')
training_set.remove('TS3012c')

test_set = ['ES2003', 'ES2004', 'ES2011', 'ES2014', 'IS1008', 'IS1009', 'TS3003', 'TS3004', 'TS3006', 'TS3007']
test_set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in test_set])

def get_data(bert=True):
    """Return (df_train_nodes, df_train_edges, df_test_nodes, df_test_edges)"""
    
    path_to_training = Path("data/training")
    path_to_test = Path("data/test")

    graph_links_labels= set()
    for id in training_set:
        with open(path_to_training / f"{id}.txt", "r") as graphe:
            for line in graphe:
                l = line.split()
                graph_links_labels.add(l[1])
    L = list(graph_links_labels)
    int2label = {indice: valeur for indice, valeur in enumerate(L)}
    label2int = {valeur: indice for indice, valeur in enumerate(L)}


    L = ['PM', 'ME', 'ID', 'UI']
    int2speaker = {indice: valeur for indice, valeur in enumerate(L)}
    speaker2int = {valeur: indice for indice, valeur in enumerate(L)}

    train_rows_nodes = [] 
    train_rows_edges = []
    test_rows_nodes = []
    test_rows_edges = []

    # extracting training data
    for k, transcription_id in enumerate(training_set):
        with open("data/training_labels.json", "r") as file:
            training_labels = json.load(file)
        #nodes
        with open(path_to_training / f"{transcription_id}.json", "r") as file:
            json_transcription = json.load(file)
        N = len(json_transcription)
        for utterance in json_transcription:
            label = training_labels[transcription_id][int(utterance["index"])]
            row_dict = {'transcription': transcription_id, 'line': int(utterance["index"]), 'speaker_int' : speaker2int[utterance["speaker"]],'speaker_text' : utterance["speaker"], 'text' : utterance["text"], 'label' : label}
            train_rows_nodes.append(row_dict)
        #edges
        with open(path_to_training / f"{transcription_id}.txt", "r") as graphe:
            for line in graphe:
                l = line.split()
                i = int(l[0])
                j =  int(l[2])
                edge_type = label2int[l[1]]
                row_dict = {'transcription': transcription_id, 'start': i, 'end' : j, 'type_int' : edge_type, 'type_text' : l[1]}
                train_rows_edges.append(row_dict)

    # extracting testing data
    for k, transcription_id in enumerate(test_set):
        #nodes
        with open(path_to_test / f"{transcription_id}.json", "r") as file:
            json_transcription = json.load(file)
        N = len(json_transcription)
        for utterance in json_transcription:
            row_dict = {'transcription': transcription_id, 'line': int(utterance["index"]), 'speaker_int' : speaker2int[utterance["speaker"]], 'speaker_text' : utterance["speaker"], 'text' : utterance["text"]}
            test_rows_nodes.append(row_dict)
        #edges
        with open(path_to_test / f"{transcription_id}.txt", "r") as graphe:
            for line in graphe:
                l = line.split()
                i = int(l[0])
                j =  int(l[2])
                edge_type = label2int[l[1]]
                row_dict = {'transcription': transcription_id, 'start': i, 'end' : j, 'type_int' : edge_type, 'type_text' : l[1]}
                test_rows_edges.append(row_dict)

    df_train_nodes = pd.DataFrame(train_rows_nodes)
    df_train_edges = pd.DataFrame(train_rows_edges)
    df_test_nodes = pd.DataFrame(test_rows_nodes)
    df_test_edges = pd.DataFrame(test_rows_edges)

    if bert:
        bert_np_list = []
        for transcription_id in training_set:
            bert_np_list.append(np.load('feature-extraction/bert/training/' + transcription_id +'.npy'))
        bert_np = np.vstack(bert_np_list)
        df_new = pd.DataFrame(bert_np, columns=['bert_' + str(k) for k in range(384)])
        df_train_nodes = pd.concat([df_train_nodes, df_new], axis=1)

        bert_np_list = []
        for transcription_id in test_set:
            bert_np_list.append(np.load('feature-extraction/bert/test/' + transcription_id +'.npy'))
        bert_np = np.vstack(bert_np_list)
        df_new = pd.DataFrame(bert_np, columns=['bert_' + str(k) for k in range(384)])
        df_test_nodes = pd.concat([df_test_nodes, df_new], axis=1)

    return df_train_nodes, df_train_edges, df_test_nodes, df_test_edges


def identity(x):
    return x

def get_graphs(nodes_features_extraction=identity, edges_features_extraction=identity, validation_ratio=0.3):
    """
    nodes_features_extraction must follow thoses rules:
        - it does not change the number of lines of the dataframe
        - it does not change the columns 'transcription', 'line' and 'label'
        - hormis ces 3 colonnes, tout sera utilisé comme feature et doit être numérique

    for edges, if df is the transformated dataframe, df['types_int'].unique().count() channels are created

    return (train_graphs, validation_graphs, test_graphs)
    """
    # getting original data
    df_train_nodes_brut, df_train_edges_brut, df_test_nodes_brut, df_test_edges_brut = get_data()
    # extracting features the same way for train and test
    df_train_nodes = nodes_features_extraction(df_train_nodes_brut)
    df_test_nodes = nodes_features_extraction(df_test_nodes_brut)
    df_train_edges = edges_features_extraction(df_train_edges_brut)
    df_test_edges = edges_features_extraction(df_test_edges_brut)

    edges_test_and_train = pd.concat([df_train_edges, df_test_edges])
    links_types = edges_test_and_train['type_int'].unique()

    train_graphs = {}
    test_graphs = {}

    for id in training_set:
        # nodes
        df_nodes = df_train_nodes[df_train_nodes['transcription'] == id]
        df_nodes = df_nodes.sort_values(by='line')
        df_nodes = df_nodes.drop('transcription', axis=1)
        df_nodes = df_nodes.drop('line', axis=1)
        labels_np = df_nodes['label'].values
        df_nodes = df_nodes.drop('label', axis=1)
        df_numerique = df_nodes.select_dtypes(include='number')
        nodes = df_numerique.to_numpy()
        x = torch.tensor(nodes, dtype=torch.float)

        # edges
        df_edges = df_train_edges[df_train_edges['transcription'] == id]
        chanels = {t : [] for t in links_types}
        for index, row in df_edges.iterrows():
            chanels[row['type_int']].append([row['start'], row['end']])
        chanels_list = chanels.values()
        edges = [torch.tensor(chanel).t().contiguous() for chanel in chanels_list]

        labels = torch.tensor(labels_np)
        graph = Data(x=x, edge_index=edges, y=labels)
        train_graphs[id] = graph

    for id in test_set:
        # nodes
        df_nodes = df_test_nodes[df_test_nodes['transcription'] == id]
        df_nodes = df_nodes.sort_values(by='line')
        df_nodes = df_nodes.drop('transcription', axis=1)
        df_nodes = df_nodes.drop('line', axis=1)
        df_numerique = df_nodes.select_dtypes(include='number')
        nodes = df_numerique.to_numpy()
        x = torch.tensor(nodes, dtype=torch.float)

        # edges
        df_edges = df_test_edges[df_test_edges['transcription'] == id]
        chanels = {t : [] for t in links_types}
        for index, row in df_edges.iterrows():
            chanels[row['type_int']].append([row['start'], row['end']])
        chanels_list = chanels.values()
        edges = [torch.tensor(chanel).t().contiguous() for chanel in chanels_list]

        labels = torch.tensor(labels_np)
        graph = Data(x=x, edge_index=edges)
        test_graphs[id] = graph
    
    def split(dic):
        liste_cles = list(dic.keys())
        random.shuffle(liste_cles)
        taille_dictionnaire_1 = int(len(liste_cles) * validation_ratio)
        sous_dictionnaire_1 = {cle: dic[cle] for cle in liste_cles[:taille_dictionnaire_1]}
        sous_dictionnaire_2 = {cle: dic[cle] for cle in liste_cles[taille_dictionnaire_1:]}
        return sous_dictionnaire_2, sous_dictionnaire_1
    
    train_graphs, validation_graphs = split(train_graphs)

    return train_graphs, validation_graphs, test_graphs




def f1_score(y_pred, y_real):
    conf_matrix = confusion_matrix(y_real, y_pred)
    tp, fp, fn, tn = conf_matrix[1, 1], conf_matrix[0, 1], conf_matrix[1, 0], conf_matrix[0, 0]
    if (tp + fp) == 0:
        return 0
    if (tp + fn) == 0:
        return 0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if (precision + recall) == 0:
        return 0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def train(model, train_graphs, optimizer, criterion,):
    """ Train graph is a dict of graph """
    model.train()
    loss_tot = 0
    for data in train_graphs.values():
        optimizer.zero_grad()  # Clear gradients.
        out = model(data)  # Perform a single forward pass.
        loss = criterion(out.float(), data.y.float().view(-1, 1))  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        loss_tot += loss
    return loss_tot


def f1_score_moyen(model, graphs):
    model.eval()
    S = 0
    for data in graphs.values():
        predicted = model.predict(data)
        f1 = f1_score(predicted, data.y.numpy())
        S += f1
    f1_moyen = S / len(graphs)
    return f1_moyen


def make_test_csv_submission(model, test_graphs, submission_name):
    test_labels = {}
    for id, graph in test_graphs.items():
        y_test = model.predict(graph)
        test_labels[id] = y_test.tolist()
    file = open("submission-"+submission_name+".csv", "w")
    file.write("id,target_feature\n")
    for key, value in test_labels.items():
        u_id = [key + "_" + str(i) for i in range(len(value))]
        target = map(str, value) 
        for row in zip(u_id, target):
            file.write(",".join(row))
            file.write("\n")
    file.close()

def analyse_model(model, validation_graphs):
    model.eval()
    S = 0
    global_conf_matrix = np.zeros((2,2))
    for data in validation_graphs.values():
        y_pred = model.predict(data)
        y_true = data.y.numpy()
        conf_matrix = confusion_matrix(y_true, y_pred)
        global_conf_matrix += conf_matrix
        f1 = f1_score(y_pred, y_true)
        S += f1
    f1_moyen = S / len(validation_graphs)
    print('-------------------------')
    print('Analyse des performance du modèle :')
    print('-------------------------')
    print('F1-score:',f1_moyen)
    plt.figure(figsize=(3, 3))
    sns.heatmap(global_conf_matrix, annot=True, cmap="Blues", cbar=False,
                xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()