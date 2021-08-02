import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def load_data_unsupervised(path="../data/cora-cocitation/", datatype="clique"):
    """
    Load citation network dataset (cora only for now)
    @param datatype:
    @type datatype:
    @param path: path to
    @type path:
    @return:
    @rtype:
    """
    print('Loading {}/{} dataset...'.format(path, datatype))

    idx_features = np.genfromtxt("{}/{}.content".format(path, datatype), dtype=np.dtype(str), skip_header=1)
    features = sp.csr_matrix(idx_features[:, 1:], dtype=np.float32)

    # build graph
    idx = np.array(idx_features[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}

    # build graph using *.cites file
    edges_unordered = np.genfromtxt("{}/{}.cites".format(path, datatype), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(features.shape[0], features.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    normalized_adj = normalize(adj + sp.eye(adj.shape[0]))

    features = torch.FloatTensor(np.array(features.todense()))
    normalized_adj = sparse_mx_to_torch_sparse_tensor(normalized_adj)

    return idx_map, adj, normalized_adj, features


def save_embeddings(output_path, embeddings, idx_map):
    """
    Write embedding vectors on a file
    @param output_path: Path of the output file.
    @type output_path: str
    @param embeddings: Matrix containing node vectors. (each row is a vector)
    @type embeddings: torch.Tensor
    @param idx_map: Dictionary that maps node id and index of the given embedding vector matrix.
    @type idx_map: dict
    """
    print("Saving embeddings...")

    id_map = {value: key for key, value in idx_map.items()}

    with open(output_path, "w") as f:
        f.write(f"{embeddings.shape[0]} {embeddings.shape[1]}\n")
        for i, emb in enumerate(embeddings):
            l = [str(e) for e in emb]
            l.insert(0, str(id_map[i]))
            f.write(" ".join(l) + "\n")


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
