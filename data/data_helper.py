import os, inspect, pickle
from time import time
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.sparse as sp


class BasicDataset(Dataset):
    def __init__(self, config:dict):
        current = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        self.d = os.path.join(current, config['data'])
        # print(f"loading [{self.d}]")

        # star expansion or clique expansion
        # self.mode_dict = {'star':0, 'clique':1}
        # self.mode      = self.mode_dict['star']
        self.IncidenceMatrix = None
        self.AdjacencyMatrix = None

        self._load_data(config)


    def _load_data(self, config:dict):
        """
        Loads the hypergraph, features, and labels.
        @return: a dictionary with hypergraph, features, and labels as keys
        @rtype: dictionary
        """
        with open(os.path.join(self.d, 'hypergraph.pickle'), 'rb') as handle:
            hypergraph = pickle.load(handle)

        with open(os.path.join(self.d, 'labels.pickle'), 'rb') as handle:
            labels = pickle.load(handle)

        split = config['split']
        with open(os.path.join(self.d, 'splits', f'{split}.pickle'), 'rb') as handle:
            s = pickle.load(handle)

        self.m_hyperedge = len(hypergraph)
        self.n_node = len(labels)
        # print(f"Total {self.n_nodes} nodes and {self.m_hyperedges} hyperedges.")

        self.hypergraph = hypergraph

        self.num_class = len(set(labels))
        self.labels = labels
        # self.labels = self._one_hot(labels, self.num_class)
        # print(f"{self.num_class} class labels.")

        self.train_idx = s['train']
        self.test_idx = s['test']
        # print(f"Training: {len(self.train_idx)} / Test: {len(self.test_idx)}")

        self.__init_features()


    def __init_features(self):
        with open(os.path.join(self.d, 'features.pickle'), 'rb') as handle:
            node_features = pickle.load(handle).todense()

        self.node_features = node_features
        self.input_dim = node_features.shape[1]

        hyperedge_features = []
        for key, value in self.hypergraph.items():
            hyperedge_feat = node_features[list(value)]
            hyperedge_feat = hyperedge_feat.mean(axis=0)
            hyperedge_features.append(hyperedge_feat)
        hyperedge_features = np.stack(hyperedge_features, axis=0)

        self.hyperedge_features = hyperedge_features


    def getAdjacencyMatrix(self):
        """
        Load or generate adjacency matrix
        @return:
        @rtype:
        """
        if self.AdjacencyMatrix is None:
            try:
                adj_mat = sp.load_npz(os.path.join(self.d, 's_pre_adj_mat.npz'))
            except:
                if self.IncidenceMatrix is None:
                    self.getIncidenceMatrix()
                I = self.IncidenceMatrix.tolil()

                print("generating adjacency matrix")
                start = time()

                # Make adjacency matrix using incidence matrix
                adj_mat = sp.dok_matrix((self.n_nodes + self.m_hyperedges, self.n_nodes + self.m_hyperedges), dtype=np.float32).tolil()
                adj_mat[:self.n_nodes, self.n_nodes:] = I
                adj_mat[self.n_nodes:, :self.n_nodes] = I.T
                adj_mat = adj_mat.tocsr()

                end = time()
                print(f"costing {end - start:.4f}s, saved adj_mat...")
                sp.save_npz(os.path.join(self.d, "s_pre_adj_mat.npz"), adj_mat)

            self.AdjacencyMatrix = adj_mat

        return self._sparse_mx_to_torch_sparse_tensor(self.AdjacencyMatrix)


    def getIncidenceMatrix(self):
        """
        Load or generate incidence matrix
        @return:
        @rtype:
        """
        if self.IncidenceMatrix is None:
            try:
                inc_mat = sp.load_npz(os.path.join(self.d, 's_pre_inc_mat.npz'))
            except:
                print("generating incidence matrix...")
                start = time()
                h_idx, deg = 0, 0
                row, col = [], []
                for node_set in self.hypergraph.values():
                    row.extend(list(node_set))
                    col.extend([h_idx for _ in range(len(node_set))])
                    h_idx += 1
                    deg += len(node_set)
                inc_mat = sp.csr_matrix((np.ones(deg), (row, col)), shape=(self.n_nodes, self.m_hyperedges))
                end = time()

                print(f"costing {end-start:.4f}s, saved inc_mat...")
                sp.save_npz(os.path.join(self.d, 's_pre_inc_mat.npz'), inc_mat)

            self.IncidenceMatrix = inc_mat

        return self._sparse_mx_to_torch_sparse_tensor(self.IncidenceMatrix)


    def getSparseGraph(self):
        if self.AdjacencyMatrix is None:
            self.getAdjacencyMatrix()
        adj_mat = self.AdjacencyMatrix
        norm_adj = self._normalize(adj_mat).tocsr()
        return self._sparse_mx_to_torch_sparse_tensor(norm_adj)


    def getFeatures(self):
        total_features = np.concatenate([self.node_features, self.hyperedge_features], axis=0)
        total_features = torch.FloatTensor(total_features)
        return torch.FloatTensor(total_features)


    def getLabels(self):
        return torch.LongTensor(self.labels)


    @property
    def n_nodes(self):
        return self.n_node


    @property
    def m_hyperedges(self):
        return self. m_hyperedge


    def _one_hot(self, labels, classes):
        """
        Converts each positive integer (representing a unique class)
        into ints one-hot form.
        @param labels: node labels
        @type labels: list of positive integers
        @return:
        @rtype:
        """
        onehot = {c: np.identity(classes)[i, :] for i, c in enumerate(range(classes))}
        return np.array(list(map(onehot.get, labels)), dtype=np.int32)


    def _normalize(self, mx):
        """
        Symmetrically normalize sparse matrix.
        @param mx: input matrix
        @type mx: scipy sparse matrix
        @return: D^{-1/2} M D^{-1/2} (where D is the diagonal node-degree matrix)
        @rtype: scipy sparse matrix
        """
        d = np.array(mx.sum(axis=1))  # D is the diagonal node-degree matrix

        d_hi = np.power(d, -0.5).flatten()
        d_hi[np.isinf(d_hi)] = 0.
        D_HI = sp.diags(d_hi)

        return D_HI.dot(mx).dot(D_HI)


    def _sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        )
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
