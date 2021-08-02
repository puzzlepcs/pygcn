import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from layers import GraphConvolution
from utils import save_embeddings, load_data_unsupervised

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train. Default is 100.')
parser.add_argument('--lr', type=float, default=0.003,
                    help='Initial learning rate. Default is 0.003.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')

parser.add_argument('--hidden_dim', type=int, default=128,
                    help='Number of hidden units. Default is 128.')
parser.add_argument('--output_dim', type=int, default=128,
                    help='Dimensionality of output vector. Default is 128.')

parser.add_argument('--input', type=str, default='../data/pubmed',
                    help='Path to input data file.')
parser.add_argument('--output', type=str, default='../emb/pubmed',
                    help='Embeddings path.')
parser.add_argument('--datatype', type=str, default='clique')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()


np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

NEG_LOSS_RATIO = 1
INTERVAL_PRINT = 5

# NUM_NODE = DATASET_NUM_DIC[args.dataset]
WEIGHT_DECAY = args.weight_decay
HIDDEN_SIZE = args.hidden_dim
EMBEDDING_SIZE = args.output_dim
LEARNING_RATE = args.lr
EPOCHS = args.epochs
DROPOUT = args.dropout

INPUT = args.input
OUTPUT = args.output
DATATYPE = args.datatype


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
        x = F.relu(self.gc1(x, normalized_adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, normalized_adj)
        emb = F.relu(x)
        return emb

    def criterion(self, x, adj):
        """
        Calculates the loss value. (Make the neighbors connected by edge close in the embedding space)
        """
        nodes = [] # list of nodes
        neighbors_list = [adj[source].nonzero()[1] for source in nodes]

        loss_total = 0

        for n in nodes:
            vector_n = x[n, :]
            neighbors = neighbors_list[n]
            num_neighbor = len(neighbors)

            if num_neighbor > 0:
                vectors_neighbors = x[neighbors, :]
                loss = -1 * torch.sum(F.logsigmoid(torch.einsum("nj, j->n"), [vectors_neighbors, vector_n]))
                loss_total += loss
        return loss_total


def main():
    # Load data
    idx_map, adj, normalized_adj, features = load_data_unsupervised(path=INPUT, datatype=DATATYPE)

    # Model and optimizer
    model = GCN(
        nfeat = features.shape[1],
        nhid = HIDDEN_SIZE,
        nout = EMBEDDING_SIZE,
        dropout = DROPOUT
    )
    print(model.train())
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, list(model.parameters())),
                                lr = LEARNING_RATE,
                                weight_decay=WEIGHT_DECAY
                )

    for epoch in range(EPOCHS + 2):
        if epoch % INTERVAL_PRINT == 0:
            t = time.time()
            model.eval()

            embeddings = model.forward(x = features, normalized_adj=normalized_adj)
            loss = model.criterion(embeddings, adj)

            optimizer.step()

            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss: {:.4f}'.format(loss),
                  'time: {:.4f}s'.format(time.time() - t))

    embeddings = embeddings.detach().numpy()
    save_embeddings(f"{OUTPUT}/{DATATYPE}.emb", embeddings, idx_map)


if __name__=="__main__":
    main()