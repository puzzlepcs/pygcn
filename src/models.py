import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from data.data_helper import BasicDataset
from src.layers import GraphConvolution

# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout):
#         super(GCN, self).__init__()
#
#         self.gc1 = GraphConvolution(nfeat, nhid)
#         self.gc2 = GraphConvolution(nhid, nclass)
#         self.dropout = dropout
#
#     def forward(self, x, adj):
#         x = F.relu(self.gc1(x, adj))
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.gc2(x, adj)
#         return F.log_softmax(x, dim=1)

#
# class MyLightGCN(nn.Module):
#     def __init__(self, num_layers, latent_dim, graph, initial_feature, num_nodes):
#         super(MyLightGCN, self).__init__()
#         self.num_nodes = num_nodes              # number of nodes
#         self.num_layers = num_layers            # number of layers
#         self.latent_dim = latent_dim            # dimensionality of embeddings
#
#         self.graph = graph                      # adjacency matrix
#         self.initial_feature = initial_feature  # initial node vector
#
#         self.embedding = nn.Embedding(
#             num_embeddings=self.num_nodes, embedding_dim=self.latent_dim)
#
#         # initialize node embedding
#         if initial_feature is None:
#             nn.init.normal_(self.embedding.weight, std=0.1)
#             print('use NORMAL distribution N(0,1) initialization')
#             # nn.init.xavier_normal_(self.embedding.weight)
#         else:
#             self.embedding.weight.data.copy_(initial_feature)
#             print('use pretrained data')
#
#         # print('graph: ', graph)
#         print('num_layers: ', self.num_layers)
#
#
#     def forward(self):
#         emb = self.embedding.weight
#
#         embs = [emb]
#
#         for layer in range(self.num_layers):
#             emb = torch.spmm(self.graph, emb)
#             embs.append(emb)
#
#         embs = torch.stack(embs, dim=1)
#         out = torch.mean(embs, dim=1)
#
#         return out

class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def topology_loss(self, neg):
        raise NotImplementedError


class LightGCN(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset: BasicDataset = dataset

        self.__init_weight()

    def __init_weight(self):
        self.num_nodes = self.dataset.n_nodes           # number of nodes
        self.num_hyperedges = self.dataset.m_hyperedges # number of hyperedges
        self.output_dim = self.dataset.num_class        # number of classes

        self.num_layers = self.config['num_layers']     # number of layers
        self.latent_dim = self.config['latent_dim']     # dimensionality of embeddings

        # Node & Hyperedge embeddings
        self.embedding_node = torch.nn.Embedding(
            num_embeddings=self.num_nodes, embedding_dim=self.latent_dim,
            device=self.config["device"]
        )
        self.embedding_hyperedge = torch.nn.Embedding(
            num_embeddings=self.num_hyperedges, embedding_dim=self.latent_dim,
            device=self.config["device"]
        )

        # initialize node embedding & hyperedge embedding
        nn.init.normal_(self.embedding_node.weight, std=0.1)
        nn.init.normal_(self.embedding_hyperedge.weight, std=0.1)

        # Final layer for classification
        self.linear_layer = torch.nn.Linear(
            in_features=self.latent_dim, out_features=self.dataset.num_class,
            bias=True, device=self.config["device"]
        )

        # Graph structure
        self.Graph = self.dataset.getSparseGraph().to(self.config["device"])
        self.Inc = torch.transpose(
            self.dataset.getIncidenceMatrix(), 0, 1
        ).to(self.config["device"])

    def computer(self):
        """
        propagation method (Similar to LightGCN)
        """
        node_emb = self.embedding_node.weight
        hyperedge_emb = self.embedding_hyperedge.weight
        all_emb = torch.cat([node_emb, hyperedge_emb])

        embs = [all_emb]
        for layer in range(self.num_layers):
            all_emb = torch.spmm(self.Graph, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        out = torch.mean(embs, dim=1)
        nodes, hyperedges = torch.split(out, [self.num_nodes, self.num_hyperedges])

        return nodes, hyperedges

    def forward(self):
        do = self.config["dropout"]
        nodes, hyperedges = self.computer()

        Z = self.linear_layer(nodes)
        Z = F.dropout(Z, do, training=self.training)

        return F.log_softmax(Z, dim=1)

    def topology_loss(self, neg):
        n, m = self.dataset.n_nodes, self.dataset.m_hyperedges
        inc = self.Inc

        num_negatives = self.config["num_negatives"]

        node_emb, hyperedge_emb = self.computer()

        total_loss = 0.0

        for idx in range(m):
            vector_src = hyperedge_emb[idx].expand(1, -1)

            pos_idx = inc[idx]._indices()[0, :]
            neg_idx = neg[idx]._indices()[0, :]

            if pos_idx.shape[0] == 0:
                continue

            vectors_pos = node_emb[pos_idx]
            vectors_neg = node_emb[neg_idx]

            pos_scores = torch.sigmoid(torch.mm(vector_src, vectors_pos.T))
            pos_scores = torch.min(pos_scores, dim=1).values

            neg_scores = torch.sigmoid(torch.mm(vector_src, vectors_neg.T))
            neg_scores = torch.reshape(neg_scores, (num_negatives, -1))
            neg_scores = torch.min(neg_scores, dim=1).values

            assert pos_scores.shape[0] * num_negatives == neg_scores.shape[0]

            scores = torch.cat([pos_scores, neg_scores])
            labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
            # scores = torch.cat([pos_scores, neg_scores], dim=1)
            # labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=1)

            loss = F.binary_cross_entropy_with_logits(scores, labels)
            total_loss += loss

        # for idx in range(m):
        #     vector_src = hyperedge_emb[idx].expand(1, -1)
        #
        #     pos_idx = torch.transpose(inc, 0, 1)[idx]._indices()[0, :]
        #     neg_idx = torch.transpose(neg, 0, 1)[idx]._indices()[0, :]
        #
        #     print(idx)
        #     print(pos_idx)
        #     print(neg_idx)
        #
        #     vectors_pos = node_emb[pos_idx]
        #     vectors_neg = node_emb[neg_idx]
        #
        #     pos_scores = torch.sigmoid(torch.mm(vector_src, vectors_pos.T))
        #     pos_scores = torch.min(pos_scores, dim=1).values
        #
        #     neg_scores = torch.sigmoid(torch.mm(vector_src, vectors_neg.T))
        #
        #     print(pos_scores)
        #     print(neg_scores)
        #
        #     neg_scores = torch.reshape(neg_scores, (num_negatives, -1))
        #     neg_scores = torch.min(neg_scores, dim=1).values
        #
        #     assert pos_scores.shape[0] * num_negatives == neg_scores.shape[0]
        #
        #     scores = torch.cat([pos_scores, neg_scores])
        #     labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        #
        #     loss = F.binary_cross_entropy_with_logits(scores, labels)
        #     total_loss += loss

        return total_loss / n


class GCN(BasicModel):
    def __init__(self,
                 config : dict,
                 dataset: BasicDataset):
        super(GCN, self).__init__()
        self.config = config
        self.dataset: BasicDataset = dataset

        self.__init_weight()

    def __init_weight(self):
        l = self.config["num_layers"]
        d, c = self.dataset.input_dim, self.dataset.num_class

        # Dimensionality of hidden dimensions
        h = [d]
        for i in range(l-1):
            power = l - i + 2
            if self.config["data"] == 'citeseer': power = l - i + 4
            h.append(2**power)
        h.append(c)

        self.layers = nn.ModuleList(
            [GraphConvolution(h[i], h[i+1]).to(self.config["device"]) for i in range(l)]
        )

        self.Graph = self.dataset.getSparseGraph().to(self.config["device"])
        self.Inc = torch.transpose(
            self.dataset.getIncidenceMatrix(), 0, 1
        ).to(self.config["device"])
        self.X = self.dataset.getFeatures().to(self.config["device"])

    def forward(self):
        do = self.config["dropout"]
        l = self.config["num_layers"]

        X = self.X
        for i, layer in enumerate(self.layers):
            X = F.relu(layer(X, self.Graph))
            if i < l-1:
                X = F.dropout(X, do, training=self.training)

        return F.log_softmax(X, dim=1)

    def computer(self):
        l = self.config["num_layers"]
        n, m = self.dataset.n_nodes, self.dataset.m_hyperedges

        X = self.X
        for i, layer in enumerate(self.layers):
            if i == l:
                break
            X = F.relu(layer(X, self.Graph))

        nodes, hyperedges = torch.split(X, [n, m])
        return nodes, hyperedges

    def topology_loss(self, neg):
        n, m = self.dataset.n_nodes, self.dataset.m_hyperedges
        inc = self.Inc
        num_negatives = self.config["num_negatives"]

        node_emb, hyperedge_emb = self.computer()

        total_loss = 0.0

        for idx in range(m):
            vector_src = hyperedge_emb[idx].expand(1, -1)

            pos_idx = inc[idx]._indices()[0, :]
            neg_idx = neg[idx]._indices()[0, :]

            if pos_idx.shape[0] == 0:
                continue

            vectors_pos = node_emb[pos_idx]
            vectors_neg = node_emb[neg_idx]

            pos_scores = torch.sigmoid(torch.mm(vector_src, vectors_pos.T))
            pos_scores = torch.min(pos_scores, dim=1).values

            neg_scores = torch.sigmoid(torch.mm(vector_src, vectors_neg.T))
            neg_scores = torch.reshape(neg_scores, (num_negatives, -1))
            neg_scores = torch.min(neg_scores, dim=1).values

            assert pos_scores.shape[0] * num_negatives == neg_scores.shape[0]

            # scores = torch.cat([pos_scores, neg_scores], dim=1)
            # labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=1)
            scores = torch.cat([pos_scores, neg_scores])
            labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])

            loss = F.binary_cross_entropy_with_logits(scores, labels)
            total_loss += loss

        # for idx in range(m):
        #     vector_src = hyperedge_emb[idx].expand(1, -1)
        #
        #     pos_idx = torch.transpose(inc, 0, 1)[idx]._indices()[0, :]
        #     neg_idx = torch.transpose(neg, 0, 1)[idx]._indices()[0, :]
        #
        #     print(idx)
        #     print(pos_idx)
        #     print(neg_idx)
        #
        #     vectors_pos = node_emb[pos_idx]
        #     vectors_neg = node_emb[neg_idx]
        #
        #     pos_scores = torch.sigmoid(torch.mm(vector_src, vectors_pos.T))
        #     pos_scores = torch.min(pos_scores, dim=1).values
        #
        #     neg_scores = torch.sigmoid(torch.mm(vector_src, vectors_neg.T))
        #
        #     print(pos_scores)
        #     print(neg_scores)
        #
        #     neg_scores = torch.reshape(neg_scores, (num_negatives, -1))
        #     neg_scores = torch.min(neg_scores, dim=1).values
        #
        #     assert pos_scores.shape[0] * num_negatives == neg_scores.shape[0]
        #
        #     scores = torch.cat([pos_scores, neg_scores])
        #     labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        #
        #     loss = F.binary_cross_entropy_with_logits(scores, labels)
        #     total_loss += loss

        return total_loss / n


class MyGCN(BasicModel):
    def __init__(self, config: dict, dataset:BasicDataset):
        super(MyGCN, self).__init__()
        self.config = config
        self.dataset = dataset

        self.__init_weight()

    def __init_weight(self):
        h = self.config["latent_dim"]
        device = self.config["device"]
        l = self.config["num_layers"]

        d, c, = self.dataset.input_dim, self.dataset.num_class

        # Layer for dimensionality reduction (d -> h)
        self.reduction_layer = torch.nn.Linear(
            in_features=d, out_features=h, bias=True, device=device
        )

        # GCN layers (h -> h)
        self.layers = nn.ModuleList(
            [GraphConvolution(h, h).to(device) for _ in range(l)]
        )

        # Layer for classsification (h -> c)
        self.classification_layer = torch.nn.Linear(
            in_features=h, out_features=h, bias=True, device=device
        )

        self.Graph = self.dataset.getSparseGraph().to(device)
        self.Inc = torch.transpose(
            self.dataset.getIncidenceMatrix(), 0, 1
        ).to(self.config["device"])
        self.X = self.dataset.getFeatures().to(self.config["device"])

    def computer(self):
        do = self.config["dropout"]
        n, m = self.dataset.n_nodes, self.dataset.m_hyperedges

        all_emb = self.X
        all_emb = self.reduction_layer(all_emb)
        all_emb = F.dropout(all_emb, do, training=self.training)

        embs = [all_emb]

        for i, layer in enumerate(self.layers):
            all_emb = F.relu(layer(all_emb, self.Graph))
            all_emb = F.dropout(all_emb, do, training=self.training)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        out = torch.mean(embs, dim=1)
        nodes, hyperedges = torch.split(out, [n, m])

        return nodes, hyperedges

    def forward(self):
        do = self.config["dropout"]

        nodes, _ = self.computer()

        Z = self.classification_layer(nodes)
        Z = F.dropout(Z, do, training=self.training)

        return F.log_softmax(Z, dim=1)

    def topology_loss(self, neg):
        n, m = self.dataset.n_nodes, self.dataset.m_hyperedges
        inc = self.Inc

        num_negatives = self.config["num_negatives"]

        node_emb, hyperedge_emb = self.computer()

        total_loss = 0.0

        for idx in range(m):
            vector_src = hyperedge_emb[idx].expand(1, -1)

            pos_idx = inc[idx]._indices()[0, :]
            neg_idx = neg[idx]._indices()[0, :]

            if pos_idx.shape[0] == 0:
                continue

            vectors_pos = node_emb[pos_idx]
            vectors_neg = node_emb[neg_idx]

            pos_scores = torch.sigmoid(torch.mm(vector_src, vectors_pos.T))
            pos_scores = torch.min(pos_scores, dim=1).values

            neg_scores = torch.sigmoid(torch.mm(vector_src, vectors_neg.T))
            neg_scores = torch.reshape(neg_scores, (num_negatives, -1))
            neg_scores = torch.min(neg_scores, dim=1).values

            assert pos_scores.shape[0] * num_negatives == neg_scores.shape[0]

            scores = torch.cat([pos_scores, neg_scores])
            labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])

            loss = F.binary_cross_entropy_with_logits(scores, labels)
            total_loss += loss

        return total_loss / n



class MyLightGCN(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(MyLightGCN, self).__init__()
        self.config = config
        self.dataset: BasicDataset = dataset

        self.__init_weight()

    def __init_weight(self):
        h = self.config["latent_dim"]
        device = self.config["device"]

        d, c = self.dataset.input_dim, self.dataset.num_class

        # Layer for dimensionality reduction
        self.reduction_layer = torch.nn.Linear(
            in_features=d, out_features=h, bias=True, device=device
        )

        # Final layer for classification
        self.classification_layer = torch.nn.Linear(
            in_features=h, out_features=c, bias=True, device=device
        )

        # Graph structure
        self.X = self.dataset.getFeatures().to(device)
        self.Graph = self.dataset.getSparseGraph().to(device)
        self.Inc = torch.transpose(self.dataset.getIncidenceMatrix(), 0, 1).to(device)

    def computer(self):
        """
        propagation method (Similar to LightGCN)
        """
        l = self.config["num_layers"]
        do = self.config["dropout"]
        n, m = self.dataset.n_nodes, self.dataset.m_hyperedges

        all_emb = self.reduction_layer(self.X)
        all_emb = F.dropout(all_emb, do, training=self.training)

        embs = [all_emb]
        for layer in range(l):
            all_emb = torch.spmm(self.Graph, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        out = torch.mean(embs, dim=1)
        nodes, hyperedges = torch.split(out, [n, m])

        return nodes, hyperedges

    def forward(self):
        do = self.config["dropout"]
        nodes, _ = self.computer()

        Z = self.classification_layer(nodes)
        Z = F.dropout(Z, do, training=self.training)

        return F.log_softmax(Z, dim=1)

    def topology_loss(self, neg):
        n, m = self.dataset.n_nodes, self.dataset.m_hyperedges
        inc = self.Inc

        num_negatives = self.config["num_negatives"]

        node_emb, hyperedge_emb = self.computer()

        total_loss = 0.0

        for idx in range(m):
            vector_src = hyperedge_emb[idx].expand(1, -1)

            pos_idx = inc[idx]._indices()[0, :]
            neg_idx = neg[idx]._indices()[0, :]

            if pos_idx.shape[0] == 0:
                continue

            vectors_pos = node_emb[pos_idx]
            vectors_neg = node_emb[neg_idx]

            pos_scores = torch.sigmoid(torch.mm(vector_src, vectors_pos.T))
            pos_scores = torch.min(pos_scores, dim=1).values

            neg_scores = torch.sigmoid(torch.mm(vector_src, vectors_neg.T))
            neg_scores = torch.reshape(neg_scores, (num_negatives, -1))
            neg_scores = torch.min(neg_scores, dim=1).values

            assert pos_scores.shape[0] * num_negatives == neg_scores.shape[0]

            scores = torch.cat([pos_scores, neg_scores])
            labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])

            loss = F.binary_cross_entropy_with_logits(scores, labels)
            total_loss += loss

        return total_loss / n

