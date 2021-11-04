# with negative sampling GCN

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from utils import save_embeddings, load_data_unsupervised

CORA_NUM_INPUT_FEATURES = OUTPUT_DIM = 128
INPUT = "../data/cora-cocitation"
OUTPUT = "../emb/cora-cocitation/gcn/neg"
DATATYPE = "clique"

INTERVAL_PRINT = 5

#
# Training params
#
def get_training_args():
    parser = argparse.ArgumentParser()

    # Training related
    parser.add_argument("--num_of_epochs", type=int, help="number of training epochs", default=200)
    parser.add_argument("--lr", type=float, help="model learning rate", default=1e-3)
    parser.add_argument("--weight_decay", type=float, help="L2 regularization on model weights", default=5e-4)

    # Dataset related
    parser.add_argument("--input", help='dataset to use for training', default="../data/cora-cocitation")
    parser.add_argument("--output", help='file to save final embeddings', default="../emb/cora-cocitation")

    args = parser.parse_args()

    # Model architecture related
    config = {
        'num_of_layers': 2,
        'num_features_per_layer': [CORA_NUM_INPUT_FEATURES, OUTPUT_DIM, OUTPUT_DIM],
        'bias': True,
        'dropout': 0.6,
        'num_negatives': 5
    }

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)

    # Add additional config information
    training_config.update(config)

    return training_config


#
# Model
#
class GCN(nn.Module):
    def __init__(self, num_of_layers:int, num_features_per_layer:list, adj, dropout=0.6, bias=True):
        super(GCN, self).__init__()
        assert num_of_layers == len(num_features_per_layer) - 1, 'Enter valid arch params.'

        layers = []
        for i in range(num_of_layers):
            layer = GraphConvolution(
                num_in_features=num_features_per_layer[i],
                num_out_features=num_features_per_layer[i+1],
                adj=adj,
                activation=nn.ELU() if i < num_of_layers - 1 else None,
                dropout_prob=dropout,
                bias=bias
            )
            layers.append(layer)

        self.gcn_net = nn.Sequential(*layers)

    def forward(self, data):
        return self.gcn_net(data)


class GraphConvolution(nn.Module):
    nodes_dim = 0
    def __init__(self, num_in_features:int, num_out_features:int, adj, activation=nn.ELU(),
                 dropout_prob=0.6, bias=True):
        super(GraphConvolution, self).__init__()
        self.num_in_features = num_in_features
        self.num_out_features = num_out_features

        self.adj = adj

        ########################################################################
        # Trainable weights: Weight matrix ("W") and bias ("b")                #
        ########################################################################
        self.weight = nn.Parameter(torch.FloatTensor(num_in_features, num_out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        ########################################################################
        # End of trainable weights                                             #
        ########################################################################
        self.leakyReLU = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.activation = activation

        self.dropout = nn.Dropout(p=dropout_prob)

        ########################################################################
        # Initialize parameters                                                #
        ########################################################################
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, in_node_features):
        #
        # STEP 1 : Linear Projection + regularization (using linear layer)
        #

        # shape = (N, FIN) where N - number of nodes in the graph, FIN - number of input features per node
        in_node_features = self.dropout(in_node_features)

        # shape = (N, FIN) * (FIN, FOUT)
        support = torch.mm(in_node_features, self.weight)

        out_node_features = torch.spmm(self.adj, support)

        if self.bias is not None:
            out_node_features = out_node_features + self.bias

        return out_node_features

    def __repr__(self):
        return self.__class__.__name__ + '(' \
                + str(self.num_in_features) + '->' \
                + str(self.num_out_features) + ')'

#
# Loss function
#
class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, node_features, adj, negatives):
        num_nodes = node_features.shape[0]

        total_loss = 0.
        for node in range(num_nodes):
            vector_source = node_features[node].expand(1, -1)

            pos_idx = adj[node]._indices()[0, :]
            neg_idx = negatives[node]._indices()[0, :]

            vectors_pos = node_features[pos_idx]
            vectors_pos = torch.transpose(vectors_pos, 0, 1)

            vectors_neg = node_features[neg_idx]
            vectors_neg = torch.transpose(vectors_neg, 0, 1)

            pos_scores = torch.mm(vector_source, vectors_pos)
            neg_scores = torch.mm(vector_source, vectors_neg)

            scores = torch.cat([pos_scores, neg_scores], dim=1)
            labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=1)

            loss = F.binary_cross_entropy_with_logits(scores, labels)
            total_loss += loss
        return total_loss / num_nodes

#
# Negative sampler
#
def uniform_sample(adj, num_negatives):
    """
    Sample negative edges for each node.
    :param adj: scipy.sparse.coo_matrix, Adjacency matrix
    :param num_negatives: int, Number of negatives to sample for each node.
    :returns: tensor.sparse.coo_tensor, Matrix containing the indices of the negative samples.
    """
    num_nodes, _ = adj.shape

    values = []
    sources = []
    targets = []
    sample_time = 0.0

    for node in range(num_nodes):
        neighbors = adj[node]._indices()[0, :]

        if len(neighbors) == 0:
            continue

        start = time.time()

        num_negative_per_node = len(neighbors) * num_negatives
        sources.extend([node] * num_negative_per_node)
        values.extend([1.0] * num_negative_per_node)

        identity = torch.zeros(num_nodes)
        identity[node] = 1.
        mask = torch.ones(num_nodes) - adj[node] - identity
        random_vector = torch.rand(num_nodes)
        random_vector = random_vector * mask

        # pick num_negative_per_node elements with largest values
        # k_th_quant = torch.topk(random_vector, num_negative_per_node, largest=False).values[-1]
        # mask = random_vector <= k_th_quant
        # negatives = mask.int().to_sparse().indices()[0,:].numpy()
        negatives = torch.topk(random_vector, num_negative_per_node, largest=True).indices
        negatives = negatives.numpy()
        if len(negatives) != num_negative_per_node:
            print(node, num_negative_per_node, len(negatives))
            print(random_vector[negatives])
            return

        targets.extend(negatives)

        sample_time += time.time() - start

    # print('Sampling time: {:.4f}s'.format(sample_time))
    try:
        s = torch.sparse_coo_tensor(indices=(sources, targets), values=values, size=(num_nodes, num_nodes))
    except:
        print(f"{len(sources)}, {len(targets)}, {len(values)}")
    return s


#
# Training
#
def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    idx_map, adj, normalized_adj, features = load_data_unsupervised(path=INPUT, datatype=DATATYPE, dim=OUTPUT_DIM)
    normalized_adj = normalized_adj.to(device)
    features = features.to(device)

    # Model and optimizer
    model = GCN(
        num_of_layers=config['num_of_layers'],
        num_features_per_layer=config['num_features_per_layer'],
        bias=config['bias'],
        dropout=config['dropout'],
        adj=normalized_adj
    ).to(device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, list(model.parameters())),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    CE = CrossEntropy()

    for epoch in range(config['num_of_epochs']):
        t = time.time()

        model.train()
        optimizer.zero_grad()

        embeddings = model.forward(features)
        negatives = uniform_sample(adj, config['num_negatives'])
        loss = CE.forward(node_features=embeddings, adj=adj, negatives=negatives)
        loss.backward()
        optimizer.step()

        if epoch % INTERVAL_PRINT == 0:
            print(
                'Epoch: {:04d}'.format(epoch + 1),
                'loss: {:.4f}'.format(loss),
                'time: {:.4f}s'.format(time.time() - t)
            )

    embeddings = model.forward(features)
    embeddings = embeddings.cpu().detach().numpy()
    save_embeddings(f"{OUTPUT}/{DATATYPE}_d{OUTPUT_DIM}.emb", embeddings, idx_map)


if __name__ == "__main__":
    config = get_training_args()
    train(config)

