# main.py - Semi-supervised node classfication

import os, inspect
from time import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim

from config.config import config
from data.data_helper import BasicDataset
from src.models import BasicModel, MyLightGCN, MyGCN
from src.utils import accuracy, sample_negatives

def train(model: BasicModel, optimizer: optim.Adam, dataset: BasicDataset):
    Y = dataset.getLabels().to(config["device"])
    train_idx = dataset.train_idx

    model.train()

    t_total = time()
    for epoch in tqdm(range(config['epochs']), desc="Training"):
        t = time()
        optimizer.zero_grad()

        Z = model.forward()

        classification_loss = F.nll_loss(Z[train_idx], Y[train_idx])
        # acc = accuracy(output=Z[train_idx], labels=Y[train_idx])

        loss = classification_loss
        if config["lambda"] != 0:
            negatives = sample_negatives(dataset, config['num_negatives']).to(config["device"])
            topology_loss = model.topology_loss(negatives)
            loss += config["lambda"] * topology_loss

        loss.backward()
        optimizer.step()

    train_time = time() - t_total
    print(' >> Optimization finished! Total elapsed time: {:.4f} s'.format(train_time))
    return train_time


def test(model: BasicModel, dataset):
    Y = dataset.getLabels().to(config["device"])

    model.eval()
    Z = model.forward()

    test_idx = dataset.test_idx
    test_loss = F.nll_loss(Z[test_idx], Y[test_idx])
    test_accuracy = accuracy(Z[test_idx], Y[test_idx])

    print("[Test set results]",
          "loss= {:.8f}".format(test_loss.item()),
          "accuracy= {:.8f}".format(test_accuracy.item()),
          "error= {:.8f}".format(100*(1-test_accuracy.item())),
          "\n")
    return test_accuracy.item()


np.random.seed(config['seed'])
torch.manual_seed(config['seed'])
if config['cuda']: torch.cuda.manual_seed(config['seed'])

# print("[data]", config['data'], "[split]", config['split'], "[n_hid]", config['latent_dim'])

# Load data
dataset = BasicDataset(config)

if config['gcn_type'] == 'gcn':
    # GCN model
    print("Aggregator: GCN")
    model = MyGCN(config=config, dataset=dataset)
else:
    # LightGCN model
    print("Aggregator: LightGCN")
    model = MyLightGCN(config=config, dataset=dataset)


# Optimizer
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, list(model.parameters())),
    lr=config['lr'],
    weight_decay=config['weight_decay']
)


# Train and Test
train_time = train(model, optimizer, dataset)
test_accuracy = test(model, dataset)

# with open('light_gcn.txt', 'a') as f:
#     f.write(f"{config['data']} {config['split']} {test_accuracy}\t{100*(1-test_accuracy)}\t{train_time}\n")

if config["save_model"]:
    current = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    d = os.path.join(current, 'result', config['data'], f"{config['data']}_lightgcn_s{config['split']}.pt")
    torch.save(model, d)
