# without negative sampling GCN
import argparse
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from constants import NumNodes, NumHyperedges
from layers import GraphConvolution
from utils import save_embeddings, load_data_unsupervised, load_hypergraph
from models import MyLightGCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train. Default is 200.')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Initial learning rate. Default is 0.001.')
parser.add_argument('--dropout', type=float, default=0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument("--early_stop", type=int, default=20)

parser.add_argument('--hidden_dim', type=int, default=128,
                    help='Number of hidden units. Default is 128.')
parser.add_argument('--output_dim', type=int, default=128,
                    help='Dimensionality of output vector. Default is 128.')

parser.add_argument('--input', type=str, default='../data/cora-cocitation',
                    help='Path to input data file.')
parser.add_argument('--output', type=str, default='../emb/cora-cocitation/gcn/',
                    help='Embeddings path.')
parser.add_argument('--datatype', type=str, default='star')

# Choose loss function
# if true, generate negative sampling and use cross entropy
parser.add_argument('--negative', type=bool, default=True)
parser.add_argument('--num_negatives', type=int, default=5)
parser.add_argument('--alpha', type=float, default=0)
parser.add_argument('--tuple_option', type=str, default='average')
parser.add_argument('--residual', type=bool, default=False)

# weather to use pretrained data
parser.add_argument('--pretrained', type=bool, default=False)

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

INTERVAL_PRINT = 1

# NUM_NODE = DATASET_NUM_DIC[args.dataset]
WEIGHT_DECAY = args.weight_decay
HIDDEN_SIZE = args.hidden_dim
EMBEDDING_SIZE = args.output_dim
LEARNING_RATE = args.lr
EPOCHS = args.epochs
DROPOUT = args.dropout
OUTPUT_DIM = args.output_dim
EARLY_STOP_PATIENT = args.early_stop

INPUT = args.input
OUTPUT = args.output
DATATYPE = args.datatype

NEGATIVE = args.negative
NUM_NEGATIVES = args.num_negatives
ALPHA = args.alpha
TUPLE_OPTION = args.tuple_option
RESIDUAL = args.residual

PRETRAINED = args.pretrained

LOG_FILE = f'log_{DATATYPE}.txt'


# Model
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.dropout = dropout

    def forward(self, x, normalized_adj):
        """
        Model Forward propagation pass.
        @param x: Arrays containing node vectors.
        @type x: torch.Tensor
        @param normalized_adj: Normalized symmetric matrix of the given graph
        @type normalized_adj: torch.Tensor (sparse matrix)
        @return: Matrix containing final embedding vectors.
        @rtype: torch.Tensor
        """
        x1 = F.relu(self.gc1(x, normalized_adj))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = self.gc2(x1, normalized_adj)
        # 마지막 layer에서는 activation function 안해줌

        if RESIDUAL:
            embs = [x1, x2]
            embs = torch.stack(embs, dim=1)
            output = torch.mean(embs, dim=1)
        else:
            output = x2

        return output

    def criterion(self, x, adj):
        """
        Calculates the loss value. (Make the neighbors connected by edge close in the embedding space)
        @param x: Embedding vectors of nodes.
        @type x: torch.Tensor
        @param adj: Adjacency matrix (each row represents the neighbors connected with edges)
        @type adj: scipy.sparse.coo_matrix
        @return: calculated loss value
        @rtype: float
        """
        num_nodes = x.shape[0]
        nodes = [n for n in range(num_nodes)]  # list of nodes

        neighbors_list = [adj[source]._indices()[0, :] for source in nodes]

        loss_total = 0.

        for n in nodes:
            vector_n = x[n, :]
            neighbors = neighbors_list[n]
            num_neighbor = len(neighbors)

            if num_neighbor > 0:
                vectors_neighbors = x[neighbors, :]
                loss = -1 * torch.sum(F.logsigmoid(torch.einsum("nj,j->n", [vectors_neighbors, vector_n])))
                loss_total += loss
        return loss_total


#
# Loss function
#
class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, node_features, adj, negatives, hyperedge_node_idx):
        num_nodes = node_features.shape[0]

        pairwise_loss = 0.
        tuplewise_loss = 0.
        for node in range(num_nodes):
            vector_source = node_features[node].expand(1, -1)

            pos_idx = adj[node]._indices()[0, :]
            neg_idx = negatives[node]._indices()[0, :]

            vectors_pos = node_features[pos_idx]
            vectors_pos = torch.transpose(vectors_pos, 0, 1)

            vectors_neg = node_features[neg_idx]
            vectors_neg = torch.transpose(vectors_neg, 0, 1)

            # Pairwise loss
            pos_scores = torch.sigmoid(torch.mm(vector_source, vectors_pos))
            neg_scores = torch.sigmoid(torch.mm(vector_source, vectors_neg))

            scores = torch.cat([pos_scores, neg_scores], dim=1)
            labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=1)

            loss = F.binary_cross_entropy_with_logits(scores, labels)
            pairwise_loss += loss

            if DATATYPE == "clique":
                continue
            # Tuplewise loss
            neg_tuplewise_scores = torch.reshape(neg_scores, (NUM_NEGATIVES, -1))

            # tuplewise loss의 option - average를 취할건지 min을 취할건지
            if TUPLE_OPTION == 'average':
                neg_tuplewise_scores = torch.mean(neg_tuplewise_scores, 1)
                pos_tuplewise_scores = torch.mean(pos_scores, dim=1)
            elif TUPLE_OPTION == 'min':
                neg_tuplewise_scores = torch.min(neg_tuplewise_scores, 1).values
                pos_tuplewise_scores = torch.min(pos_scores, dim=1).values

            assert pos_tuplewise_scores.shape[0] * NUM_NEGATIVES == neg_tuplewise_scores.shape[0]

            scores = torch.cat([pos_tuplewise_scores, neg_tuplewise_scores])
            labels = torch.cat([torch.ones_like(pos_tuplewise_scores), torch.zeros_like(neg_tuplewise_scores)])

            loss = F.binary_cross_entropy_with_logits(scores, labels)
            tuplewise_loss += loss

        pairwise_loss = pairwise_loss / num_nodes # node 갯수만큼 나눠줌
        tuplewise_loss = 0 if DATATYPE == "clique" else tuplewise_loss / (num_nodes - hyperedge_node_idx) # hyperedge 의 갯수만큼 나눠줌
        return pairwise_loss + tuplewise_loss
        # return pairwise_loss

#
# Negative sampler
#
def uniform_sample(adj):
    """
    Sample negative edges for each node.
    :param adj: scipy.sparse.coo_matrix, Adjacency matrix
    :returns: tensor.sparse.coo_tensor, Matrix containing the negative samples(negative edges).
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

        num_negative_per_node = len(neighbors) * NUM_NEGATIVES
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
# Negative sampler for star expansion
#
def uniform_sample_star(adj, hyperedge_node_idx):
    num_nodes, _ = adj.shape

    values = []
    sources = []
    targets = []

    for node in range(num_nodes):
        neighbors = adj[node]._indices()[0, :]

        if len(neighbors) == 0:
            continue

        num_negative_per_node = len(neighbors) * NUM_NEGATIVES
        sources.extend([node] * num_negative_per_node)
        values.extend([1.0] * num_negative_per_node)

        if node < hyperedge_node_idx:
            # Node type: Node
            node_mask      = torch.ones(hyperedge_node_idx)
            hyperedge_mask = torch.zeros(num_nodes - hyperedge_node_idx)
        else:
            # Node type: Hyperedge
            node_mask      = torch.zeros(hyperedge_node_idx)
            hyperedge_mask = torch.ones(num_nodes - hyperedge_node_idx)
        type_mask = torch.cat((node_mask, hyperedge_mask))
        assert type_mask.shape[0] == num_nodes

        # mask prevents from sampling nodes which are connected by an edge or the same type
        mask = torch.ones(num_nodes) - adj[node] - type_mask
        random_vector = torch.rand(num_nodes)
        random_vector = random_vector * mask

        # pick num_negative_per_node elements with largest values
        negatives = torch.topk(random_vector, num_negative_per_node, largest=True).indices
        negatives = negatives.numpy()
        if len(negatives) != num_negative_per_node:
            print(node, num_negative_per_node, len(negatives))
            print(random_vector[negatives])
            return

        targets.extend(negatives)

    print(len(values))
    try:
        s = torch.sparse_coo_tensor(indices=(sources, targets), values=values, size=(num_nodes, num_nodes))
    except:
        print(f"{len(sources)}, {len(targets)}, {len(values)}")
    return s



def main():
    param_str = '''
[PARAMS]
  data: {}\n  expansion: {}\n  dim: {}\n  epochs: {}\n  lr: {}
  dropout: {} \n  negative: {}\n  alpha: {}
  tuple_option: {}\n  residual: {}\n  pretrained: {}'''.format(
        INPUT, DATATYPE, OUTPUT_DIM, EPOCHS, LEARNING_RATE, DROPOUT, NEGATIVE, ALPHA, TUPLE_OPTION, RESIDUAL, PRETRAINED
    )
    print(param_str)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    idx_map, adj, normalized_adj, features = load_data_unsupervised(path=INPUT, datatype=DATATYPE, dim=OUTPUT_DIM)

    # hypergraph = load_hypergraph(path=INPUT)
    normalized_adj = normalized_adj.to(device)
    # adj = adj.to(device)

    dataset = os.path.split(args.input)[-1]

    if dataset == "dblp":
        num_nodes = NumNodes.DBLP
        num_hyperedges = NumHyperedges.DBLP
    elif dataset == "pubmed":
        num_nodes = NumNodes.PUBMED
        num_hyperedges = NumHyperedges.PUBMED
    else:
        num_nodes = NumNodes.CORA
        num_hyperedges = NumHyperedges.CORA

    if DATATYPE == 'star':
        total_nodes = num_nodes + num_hyperedges
    else:
        total_nodes = num_nodes

    if PRETRAINED:
        features = features.to(device)
    else:
        features = None
        # features = torch.normal(mean=1.0, std=0.1, size=(total_nodes, HIDDEN_SIZE))
        # features = features.to(device)

    # File stream for logging
    train_log = open(os.path.join(OUTPUT, LOG_FILE), 'a', newline='')
    train_log.write(param_str)


    # Model and optimizer
    # model = GCN(
    #     nfeat=features.shape[1],
    #     nhid=HIDDEN_SIZE,
    #     nout=EMBEDDING_SIZE,
    #     dropout=DROPOUT
    # ).to(device)

    # LightGCN model
    model = MyLightGCN(
        num_layers=2,
        latent_dim=HIDDEN_SIZE,
        graph=adj.to(device),
        # graph=normalized_adj,
        initial_feature=features,
        num_nodes=total_nodes
    )

    model = model.to(device)
    print('>> Model parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, list(model.parameters())),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    if args.negative:
        CE = CrossEntropy()

    # Model training start
    wait = 0
    loss_min = np.inf

    print('Start training...\n')
    for epoch in range(EPOCHS):
        if wait >= EARLY_STOP_PATIENT:
            earlystop_log = 'Early stop at epoch :%04d' % (epoch)
            print(earlystop_log)
            train_log.write(earlystop_log)
            break

        t = time.time()

        model.train()
        optimizer.zero_grad()

        # embeddings = model.forward(x=features, normalized_adj=normalized_adj)
        embeddings = model.forward()

        if NEGATIVE:
            if DATATYPE == 'clique':
                negatives = uniform_sample(adj)
            else:
                negatives = uniform_sample_star(adj, num_nodes)
            loss = CE.forward(node_features=embeddings, adj=adj, negatives=negatives, hyperedge_node_idx=num_nodes)
        else:
            loss = model.criterion(embeddings, adj)

        loss.backward()
        # for p in model.parameters():
        #     print(p)
        optimizer.step()

        epoch_t = time.time() - t
        train_log_str = '[Epoch {:03d}/{:03d}] - loss: {:.8f} time: {:.4f}s\n'.format(epoch + 1, EPOCHS, loss, epoch_t)
        train_log.write(train_log_str)

        if epoch % INTERVAL_PRINT == 0:
            print(train_log_str, end='')

        if loss <= loss_min:
            wait = 0
            loss_min = loss
        else:
            wait += 1

    model.eval()
    # embeddings = model.forward(features, normalized_adj)
    embeddings = model.forward()
    embeddings = embeddings.cpu().detach().numpy()
    save_embeddings(f"{OUTPUT}/{DATATYPE}_d{OUTPUT_DIM}.emb", embeddings, idx_map)

    train_log.write('\n\n')


if __name__ == "__main__":
    main()
