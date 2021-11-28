import stellargraph as sg
import pandas as pd
import numpy as np
import networkx as nx
import tensorflow as tf
from tensorflow import keras
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import DeepGraphCNN, GCNSupervisedGraphClassification
from stellargraph import StellarGraph
from stellargraph import datasets
from stellargraph.datasets.dataset_loader import DatasetLoader
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn import model_selection
from sklearn import preprocessing

import argparse
parser = argparse.ArgumentParser(description='graph embedding karate')
parser.add_argument('-p', "--percentage", dest='percentage', metavar='<percentage>', type=int, help='attack percentage (default 0)', default=0, required=False)
parser.add_argument('-c', "--criteria", dest='criteria', metavar='<criteria>', type=str, help='attack criteria (default random)', default='random', choices=['random', 'betweeness'], required=False)
parser.add_argument('-i', "--path", dest='path', metavar='<input-path>', type=str, help='input directory (default .)', default='.', required=False)
parser.add_argument('-f', "--confpath", dest='confpath', metavar='<conf-path>', type=str, help='input directory (default .)', default='.', required=False)
parser.add_argument('-o', "--outpath", dest='outpath', metavar='<out-path>', type=str, help='out directory (default .)', default='.', required=False)
parser.add_argument('-t', "--ontest", action='store_true', default=False, required=False)
parser.add_argument('-d', "--dataset", dest='dataset', metavar='<dataset>', type=str, help='dataset name (default MUTAG)', choices=['MUTAG', 'PROTEINS', 'KIDNEY', 'Kidney_9.2', 'KIDNEYdeg'], default='MUTAG', required=False)
parser.add_argument('-V', "--verbose", action='store_true', required=False)
parser.add_argument('-N', "--nxmode", action='store_true', required=False)


# The edge attack routine
import networkx as nx
import operator
import random
class KIDNEY(
    DatasetLoader,
    name="KIDNEY",
    directory_name="KIDNEY",
    url="file:///Users/maurizio/Google Drive/Software/graph_nn/data/KIDNEY.zip",
    url_archive_format="zip",
    expected_files=[
        "KIDNEY_A.txt",
        "KIDNEY_graph_indicator.txt",
        "KIDNEY_node_labels.txt",
        "KIDNEY_edge_labels.txt",
        "KIDNEY_graph_labels.txt",
        "KIDNEY.txt",
    ],
    description="Some description.'"
    "The dataset includes 299 graphs with 1034 nodes and.",
    source="file:///Users/maurizio/Google Drive/TUDatasets/KIDNEY.zip",
):
    _edge_labels_as_weights = True
    _node_attributes = True

    def load(self):
        """
        """
        return load_graph_kernel_dataset(self)

class MUTAG(
    DatasetLoader,
    name="MUTAG",
    directory_name="MUTAG",
    url="file:///Users/maurizio/stellargraph/MUTAG.zip",
    url_archive_format="zip",
    expected_files=[
        "MUTAG_A.txt",
        "MUTAG_graph_indicator.txt",
        "MUTAG_node_labels.txt",
        "MUTAG_edge_labels.txt",
        "MUTAG_graph_labels.txt",
        "MUTAG.txt",
    ],
    description="Some description.'"
    "The dataset includes 299 graphs with 1034 nodes and.",
    source="file:///Users/maurizio/stellargraph/MUTAG.zip",
):
    _edge_labels_as_weights = False
    _node_attributes = False

    def load(self):
        """
        """
        return load_graph_kernel_dataset(self)

class PROTEINS(
    DatasetLoader,
    name="PROTEINS",
    directory_name="PROTEINS",
    url="file:///Users/maurizio/stellargraph/PROTEINS.zip",
    url_archive_format="zip",
    expected_files=[
        "PROTEINS_A.txt",
        "PROTEINS_graph_indicator.txt",
        "PROTEINS_node_labels.txt",
        "PROTEINS_node_attributes.txt",
        "PROTEINS_edge_labels.txt",
        "PROTEINS_graph_labels.txt",
        "PROTEINS.txt",
    ],
    description="Some description.'"
    "The dataset includes 299 graphs with 1034 nodes and.",
    source="file:///Users/maurizio/stellargraph/PROTEINS.zip",
):
    _edge_labels_as_weights = False
    _node_attributes = True

    def load(self):
        """
        """
        return load_graph_kernel_dataset(self)

def load_graph_kernel_dataset(dataset):

    #dataset.download()

    def _load_from_txt_file(filename, names=None, dtype=None, index_increment=None):
        df = pd.read_csv(
            dataset._resolve_path(filename=f"{dataset.name}_{filename}.txt"),
            header=None,
            index_col=False,
            dtype=dtype,
            names=names,
        )
        # We optional increment the index by 1 because indexing, e.g. node IDs, for this dataset starts
        # at 1 whereas the Pandas DataFrame implicit index starts at 0 potentially causing confusion selecting
        # rows later on.
        if index_increment:
            df.index = df.index + index_increment
        return df

    # edge information:
    df_graph = _load_from_txt_file(filename="A", names=["source", "target"])

    if dataset._edge_labels_as_weights:
        # there's some edge labels, that can be used as edge weights
        df_edge_labels = _load_from_txt_file(
            filename="edge_labels", names=["weight"], dtype=int
        )
        df_graph = pd.concat([df_graph, df_edge_labels], axis=1)

    # node information:
    df_graph_ids = _load_from_txt_file(
        filename="graph_indicator", names=["graph_id"], index_increment=1
    )

    df_node_labels = _load_from_txt_file(
        filename="node_labels", dtype="category", index_increment=1
    )
    # One-hot encode the node labels because these are used as node features in graph classification
    # tasks.
    df_node_features = pd.get_dummies(df_node_labels)

    if dataset._node_attributes:
        # there's some actual node attributes
        df_node_attributes = _load_from_txt_file(
            filename="node_attributes", dtype=np.float32, index_increment=1
        )

        df_node_features = pd.concat([df_node_features, df_node_attributes], axis=1)

    # graph information:
    df_graph_labels = _load_from_txt_file(
        filename="graph_labels", dtype="category", names=["label"], index_increment=1
    )

    # split the data into each of the graphs, based on the nodes in each one
    def graph_for_nodes(nodes):
        # each graph is disconnected, so the source is enough to identify the graph for an edge
        edges = df_graph[df_graph["source"].isin(nodes.index)]
        return StellarGraph(nodes, edges)

    groups = df_node_features.groupby(df_graph_ids["graph_id"])
    graphs = [graph_for_nodes(nodes) for _, nodes in groups]

    return graphs, df_graph_labels["label"]

def nx_edgeattack(G: nx.Graph, criteria = "random", percentage=30, verbose=False, random_state=42):
  at = percentage/100.0
  #remove_zero_weights(G)
  if criteria == "betweeness":
    score = nx.edge_betweenness(G).items()
  elif criteria == "degree":
    raise Exception("Wrong criteria")
  elif criteria == "random":
    score = list(G.edges())
    random.Random(random_state).shuffle(score)
    score = list(dict(zip(score,range(len(score)))).items())
  else:
    raise Exception("Wrong criteria")
  edges_to_remove = sorted(score, key=operator.itemgetter(1, 0), reverse=True)[0:int(len(score)*at)]
  #assert len(edges_to_remove) > 0, "Nothing to remove!"
  for e,w in edges_to_remove:
    G.remove_edge(e[0], e[1])
  if verbose:
    print("removed", edges_to_remove)
  return 0,len(edges_to_remove)

# The loading function
def load_graphs(input_path, dataname, fmt='graphml', ontest=True, percentage=20, criteria='random'):
      datapath = f'{input_path}/{dataname}/{fmt}'
      if not os.path.isdir(datapath):
        raise Exception(f'Wrong input path! {datapath}')
      filenames = os.listdir(datapath)
      print("Loading " + dataname + " graphs with networkx...")
      graphs = []
      graphsadv = []
      nxgraphs = []
      targets = []
      dfl = pd.read_csv(f'{input_path}/{dataname}/{dataname}.txt', sep='\t')
      last_column = dfl.iloc[:,[0] + [-1]]
      labelset = set()
      for file in tq.tqdm(last_column['Samples'].values):
            if fmt=='graphml':
                G = nx.read_graphml(os.path.join(datapath,f'{file}.{fmt}'))
            else:
                G = nx.read_edgelist(os.path.join(datapath,f'{file}.{fmt}'))
            nxgraphs.append(G)
            targets.append(last_column[last_column['Samples'].astype(str) == file].iloc[:,-1:])
            data=nx.get_node_attributes(G,'label')
            labelset = labelset.union(set(data.values()))
      labels = list(labelset)
      #print("LABELS ", labels)
      if len(labels) > 0:   # if there are node labels ... use degree (otherwise not working)
        for G in nxgraphs:
              for l in labels:
                G.add_node(f'dummy-{l}', label=l)
              df = pd.DataFrame(G.nodes(data='label'), columns=['id','label'])
              df = df.set_index('id')
              for l in labels:
                G.remove_node(f'dummy-{l}')
              node_features = pd.get_dummies(df['label'], prefix='label')
              node_features = node_features[:-(len(labels))]
              # copy graphs and attack one set 
              Gadv = G.copy()
              if percentage > 0: 
                ec = G.number_of_edges()
                n,e = nx_edgeattack(Gadv, criteria=criteria, percentage=percentage, random_state=42)
              graphs += [sg.StellarGraph.from_networkx(G,node_type_default="default", edge_type_default="default", node_type_attr="type", edge_type_attr="type", edge_weight_attr="label", node_features=node_features)]
              graphsadv += [sg.StellarGraph.from_networkx(Gadv,node_type_default="default", edge_type_default="default", node_type_attr="type", edge_type_attr="type", edge_weight_attr="label", node_features=node_features)]
      else:       # otherwise node labels are degrees
        for G in nxgraphs:
              df = pd.DataFrame(G.degree(), columns=['id','degree'])
              df = df.set_index('id')
              node_features = pd.get_dummies(df['degree'], prefix='degree')
              Gadv = G.copy()
              if percentage > 0: 
                ec = G.number_of_edges()
                n,e = nx_edgeattack(Gadv, criteria=criteria, percentage=percentage, random_state=42)
              graphs += [sg.StellarGraph.from_networkx(G,node_type_default="default", edge_type_default="default", node_type_attr="type", edge_type_attr="type", edge_weight_attr="label", node_features=node_features)]
              graphsadv += [sg.StellarGraph.from_networkx(Gadv,node_type_default="default", edge_type_default="default", node_type_attr="type", edge_type_attr="type", edge_weight_attr="label", node_features=node_features)]
      if percentage > 0: print("Removed %d nodes (over %d) and %d edges (over %d) in last graph!"%(n,G.number_of_nodes(),e,ec))
      #print(filenames[0].split('.')[0])
      from sklearn import preprocessing
      le = preprocessing.LabelEncoder()
      le.fit(np.ravel(targets))
      y = le.transform(np.ravel(targets))
      df_labels = pd.DataFrame()
      df_labels['label'] = np.array(y)
      df_labels.index = df_labels.index + 1  # index increment
      print(df_labels.value_counts())
      if ontest:
        return graphs,graphsadv,df_labels['label']
      else:
        return graphsadv,graphs,df_labels['label']

# The GCNN model
def create_dgcnn_model(generator, size, nouts, learnrate=0.0001):
      k = 35  # the number of rows for the output tensor
      layer_sizes = [size, size, size, nouts]
      dgcnn_model = DeepGraphCNN(layer_sizes=layer_sizes,activations=["tanh", "tanh", "tanh", "tanh"],k=k,bias=False,generator=generator)
      x_inp, x_out = dgcnn_model.in_out_tensors()
      x_out = Conv1D(filters=16, kernel_size=sum(layer_sizes), strides=sum(layer_sizes))(x_out)
      x_out = MaxPool1D(pool_size=2)(x_out)
      x_out = Conv1D(filters=32, kernel_size=5, strides=1)(x_out)
      x_out = Flatten()(x_out)
      x_out = Dense(units=size, activation="relu")(x_out)
      x_out = embedlayer = Dropout(rate=0.5)(x_out)
      if nouts > 2:
        predictions = Dense(units=nouts, activation="softmax")(x_out)
        model = Model(inputs=x_inp, outputs=predictions)
        model.compile(optimizer=Adam(lr=learnrate), loss=categorical_crossentropy, metrics=["acc"],)
      else:
        predictions = Dense(units=1, activation="sigmoid")(x_out)
        model = Model(inputs=x_inp, outputs=predictions)
        model.compile(optimizer=Adam(lr=learnrate), loss=binary_crossentropy, metrics=["acc"],)
      embedding = Model(inputs=x_inp, outputs=embedlayer)
      return model, embedding

# Load the dataset
import os
import pandas as pd
import numpy as np
import random
import operator
import tqdm as tq
#@title  { form-width: "30%" }

args = parser.parse_args()
dataset = args.dataset #@param ['MUTAG', 'KIDNEY', 'Kidney_9.2', 'PROTEINS', 'JE']
criteria = args.criteria #@param ['random', 'betweeness']
percentage = args.percentage #@param ["0.5", "1.0", "5.0", "10", "20", "30", "40", "50", "0"] {type:"raw"}
ontest = args.ontest #@param {type:"boolean"}
path = args.path #@param {type:"string"}
if args.nxmode:
  graphs, graphsadv, graph_labels = load_graphs(path,dataset, criteria=criteria, percentage=percentage, ontest=ontest)
else:
  data = globals()[dataset]()
  graphs, graph_labels = data.load()
  graphsadv = []  # no attack if dataset is loaded from TU format
  for G in tq.tqdm(graphs, desc="Copy/Attack"):
    Gadv = G.to_networkx()
    ec = Gadv.number_of_edges()
    if percentage > 0: 
      n,e = nx_edgeattack(Gadv, criteria=criteria, percentage=percentage, random_state=42)
    graphsadv += [sg.StellarGraph.from_networkx(Gadv,node_type_default="default", edge_type_default="default", node_type_attr="type", edge_type_attr="type", edge_weight_attr="weight", node_features='feature')]
if graph_labels.nunique() > 2:
  y = pd.get_dummies(graph_labels)
else:
  y = pd.get_dummies(graph_labels, drop_first=True)
print(graphs[0].node_features())
print(graphs[0].info())
print(y)
nclasses = len(y.columns)
print("No Classes %d\n"%nclasses)
summary = pd.DataFrame(
    [(g.number_of_nodes(), g.number_of_edges(),ga.number_of_nodes(), ga.number_of_edges()) for g,ga in zip(graphs,graphsadv)],
    columns=["(Graph) nodes", "(Graph) edges", "(Attacked Graph) nodes", "(Attacked Graph) edges"],
)
print(summary.describe().round(2))

# Load model parameters
import os
import json
method='DGCNN'
confpath = args.confpath #@param {type:"string"}
path = os.path.join(confpath, f'{method}_{dataset}_params.json')
if os.path.isfile(path):
  params = json.load( open( path, 'r' ) )
  print(params)
else:
  print("No default found!")

# Validation
#@title  { form-width: "30%" }
import warnings
warnings.filterwarnings("ignore")
from time import time
start = time()
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,matthews_corrcoef,accuracy_score,precision_score,f1_score, recall_score
import tqdm as tq
tot_preds = np.array([])
tot_targets = np.array([])
tot_acc = np.array([])
tot_prec = np.array([])
tot_F1 = np.array([])
tot_recall = np.array([])
tot_MCC = np.array([])
cv_folds = 10
tsize = 1.0 - (1.0 / float(cv_folds))
test_metrics = []
verbose = 1 if params['verbose'] else 0
cv_folds = 10 #@param {type:"slider", min:2, max:10, step:1}
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
setlist = []
#for i in tq.tqdm(range(cv_folds), desc="fold: "):
#  train_graphs, test_graphs = model_selection.train_test_split(y, train_size=0.9, test_size=None, stratify=y)
#  train_graphs = pd.DataFrame(data=train_graphs)
#  test_graphs = pd.DataFrame(data=test_graphs)
#  gen = PaddedGraphGenerator(graphs=graphs)
#  genadv = PaddedGraphGenerator(graphs=graphsadv)
#  train_gen = gen.flow(list(train_graphs.index - 1),targets=train_graphs.values,batch_size=1,symmetric_normalization=False)
#  test_gen = genadv.flow(list(test_graphs.index - 1),targets=test_graphs.values,batch_size=1,symmetric_normalization=False)
if nclasses > 2:
  yy = np.argmax(y.values, axis=1)
  y = y.to_numpy()
else:
  y = y.to_numpy()
  yy = y
#print(type(y),y)
for train_index, test_index in tq.tqdm(list(skf.split(graphs,yy)), desc="fold: "):
  train_graphs = graph_labels[train_index+1]
  test_graphs = graph_labels[test_index+1]
  setlist += [set(test_index)]
  gen = PaddedGraphGenerator(graphs=graphs)
  genadv = PaddedGraphGenerator(graphs=graphsadv)
  train_gen = gen.flow(list(train_graphs.index - 1),targets=y[np.array(train_graphs.index-1)],batch_size=1,symmetric_normalization=False)
  test_gen = genadv.flow(list(test_graphs.index - 1),targets=y[np.array(test_graphs.index-1)],batch_size=1,symmetric_normalization=False)
  y_train = [x.tolist() for x in y[np.array(train_graphs.index-1)]] 
  y_test = [x.tolist() for x in y[np.array(test_graphs.index-1)]]
  #model, embedding = create_gnn_model(generator,args.layerdim, nclasses, learnrate=args.learningrate)
  #if nclasses > 2:
    #y_train = train_graphs.idxmax(axis=1).values
    #y_test = test_graphs.idxmax(axis=1).values
  #else:
  #  y_train = train_graphs.values.ravel()
  #  y_test = test_graphs.values.ravel()
  print(type(y_test), y_test)
  model, embedding = create_dgcnn_model(gen, params['layerdim'], nclasses, learnrate=params['learningrate'])
  history = model.fit(train_gen, validation_data=test_gen, shuffle=False, epochs=params['epochs'], verbose=params['verbose'])
  X_test = embedding.predict(test_gen)
  X_train = embedding.predict(train_gen)
  y_pred = SVC(kernel='linear').fit(X_train,y_train).predict(X_test)
  tot_preds = np.append(tot_preds,y_pred)
  tot_targets = np.append(tot_targets,y_test)
  tot_acc = np.append(tot_acc, accuracy_score(y_test, y_pred))
  tot_prec = np.append(tot_prec, precision_score(y_test, y_pred, average='macro'))
  tot_F1 = np.append(tot_F1, f1_score(y_test, y_pred, average='macro'))
  tot_recall = np.append(tot_recall, recall_score(y_test, y_pred, average='macro'))
  tot_MCC = np.append(tot_MCC, matthews_corrcoef(y_test, y_pred))
temp = time() - start
hours = temp//3600
temp = temp - 3600*hours
minutes = temp//60
seconds = temp - 60*minutes
expired = '%d:%d:%d' %(hours,minutes,seconds)
print()
print(confusion_matrix(tot_targets, tot_preds))
print("Acc\t%.2f\u00B1%.2f"%((tot_acc * 100).mean(), (tot_acc * 100).std()))
print("Prec\t%.2f\u00B1%.2f"%(tot_prec.mean(), tot_prec.std()))
print("F1\t%.2f\u00B1%.2f"%(tot_F1.mean(), tot_F1.std()))
print("Recall\t%.2f\u00B1%.2f"%(tot_recall.mean(), tot_recall.std()))
print('MCC\t%.2f\u00B1%.2f'%(tot_MCC.mean(), tot_MCC.std()))
print(set.intersection(*setlist))
import sys
# Saving results
method = 'DGCNNtu'
#@title  { form-width: "30%" }
outpath = args.outpath #@param {type:"string"}
from datetime import datetime
import pandas as pd
path = os.path.join(outpath, f'{method}_{dataset}_e{params["epochs"]}_l{params["layerdim"]}.csv')
if not os.path.exists(path):
  dfres = pd.DataFrame(columns=['mode', 'criteria', '% attack','avg edge del', 'acc','prec','f1','recall','MCC','cm', 'date', 'time'])
  dfres.to_csv(path, index=False)
dfres = pd.read_csv(path)
mode = 'test' if ontest else 'train'
dfres = dfres.append({'mode' : mode, 'criteria' : criteria, '% attack': str(percentage), 
                      'avg edge del' : "%.2f"%(abs(float(summary.describe().iat[1,1]) - float(summary.describe().iat[1,3]))),
                      'acc' : "%.2f\u00B1%.2f"%(tot_acc.mean(), tot_acc.std()), 
                      'prec' : "%.2f\u00B1%.2f"%(tot_prec.mean(), tot_prec.std()),
                      'f1' : "%.2f\u00B1%.2f"%(tot_F1.mean(), tot_F1.std()),
                      'recall' : "%.2f\u00B1%.2f"%(tot_recall.mean(), tot_recall.std()),
                      'MCC' : "%.2f\u00B1%.2f"%(tot_MCC.mean(), tot_MCC.std()),
                      'cm' : f'{confusion_matrix(tot_targets, tot_preds)}'.replace('\n',''),
                      'date': datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                      'time' : expired}, ignore_index=True)
dfres.to_csv(path, index=False)
print(dfres)
