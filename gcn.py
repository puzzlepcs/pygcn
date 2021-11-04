# gcn on star

import os, inspect
from time import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim

from config.config import config
from data.data_helper import BasicDataset
from src.models import MyGCN
from src.utils import accuracy, sample_negatives


def train(model: MyGCN, optimizer: optim.Adam, dataset:BasicDataset):
    Y = dataset.getLabels().to(config["device"])

    train_idx = dataset.train_idx

    model.train()

    t_total = time()
    for epoch in tqdm(range(config["epochs"])):
        t = time()
        optimizer.zero_grad()

        Z = model.forward()

        classification_loss = F.nll_loss(Z[train_idx], Y[train_idx])
        acc = accuracy(Z[train_idx], Y[train_idx])

        loss = classification_loss
        if config["lambda"] != 0:
            negatives = sample_negatives(dataset, config["num_negatives"]).to(config["device"])
            topology_loss = model.topology_loss(negatives)

            loss += config["lambda"] * topology_loss

        loss.backward()
        optimizer.step()

        # if epoch % 20 == 0:
        #     print('[Epoch {:4d}]'.format(epoch + 1),
        #           'loss= {:.8f}'.format(loss.item()),
        #           'accuracy= {:.8f}'.format(acc.item()),
        #           'time= {:.4f}s'.format(time() - t))

    train_time = time() - t_total
    print(' >> Optimization finished! Total elapsed time: {:.4f} sec'.format(train_time))
    return train_time


def test(model, dataset):
    Y = dataset.getLabels().to(config["device"])

    model.eval()
    Z = model.forward()

    test_idx = dataset.test_idx
    test_loss = F.nll_loss(Z[test_idx], Y[test_idx])
    test_accuracy = accuracy(Z[test_idx], Y[test_idx])

    print("[Test set results]",
          "loss= {:.8f}".format(test_loss.item()),
          "accuracy= {:.8f}".format(test_accuracy.item()),
          "error= {:.8f}".format(100 * (1 - test_accuracy.item())),
          "\n")
    return test_accuracy.item()

np.random.seed(config["seed"])
torch.manual_seed(config["seed"])
if config["cuda"]: torch.cuda.manual_seed(config["seed"])

# print("[data]", config['data'], "[split]", config['split'], "[n_hid]", config['latent_dim'])

# Load data
dataset = BasicDataset(config)


# GCN model
model = MyGCN(config=config, dataset=dataset)
# print(model)


# Optimizer
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, list(model.parameters())),
    lr=config["lr"], weight_decay=config["weight_decay"]
)


# Train & Test
train_time = train(model, optimizer, dataset)
test_accuracy = test(model, dataset)

# with open('gcn.txt', 'a') as f:
#     f.write(f"{test_accuracy}\t{100*(1-test_accuracy)}\t{train_time}\n")

if config["save_model"]:
    current = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    d = os.path.join(current, 'result', config['data'], f"{config['data']}_gcn_s{config['split']}.pt")
    torch.save(model, d)