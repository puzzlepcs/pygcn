# config.py

'''
data: dataset
datatype: clique / star
split: 1-10
'''
data = 'cora-coauthorship'
datatype = 'star'
split = 1


'''
Random seed
'''
seed = 42


'''
model related parameters
'''
# tuple = 'min'
# gcn_type = 'gcn'
# skip_connection = True
layers = 2
epochs = 200
latent_dim = 16
save_model = True


'''
parameters of optimization (HyperGCN과 동일)
'''
learning_rate = 1e-2
weight_decay = 5e-4
dropout = 0.5
lambda_ = 1


import argparse, torch

parser = argparse.ArgumentParser()


parser.add_argument('--seed', type=int, default=seed, help='Random seed.')


parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--save-model', action='store_true', default=False)


# Data settings
parser.add_argument('--data', type=str, default=data,
                    help='Path to input data file.')
parser.add_argument('--datatype', type=str, default=datatype)
parser.add_argument('--split', type=str, default=split)


# Training settings
parser.add_argument('--epochs', type=int, default=epochs,
                    help=f'Number of epochs to train. Default is {epochs}.')
parser.add_argument('--lr', type=float, default=learning_rate,
                    help=f'Initial learning rate. Default is {learning_rate}.')
parser.add_argument('--dropout', type=float, default=dropout,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight-decay', type=float, default=weight_decay,
                    help='Weight decay (L2 loss on parameters).')

parser.add_argument('--latent-dim', type=int, default=latent_dim,
                    help=f'Number of hidden units. Default is {latent_dim}.')

parser.add_argument('--lambda', type=float, default=lambda_)


# Model related settings
# parser.add_argument('--tuple_option', type=str, default=tuple)
# parser.add_argument('--gcn_type', type=str, default=gcn_type)
# parser.add_argument('--skip_connection', type=bool, default=skip_connection)
parser.add_argument('--num-negatives', type=int, default=5)
parser.add_argument('--num-layers', type=int, default=layers)


args = parser.parse_args()

config = dict()
for arg in vars(args):
    config[arg] = getattr(args, arg)

config["cuda"] = not config["no_cuda"] and torch.cuda.is_available()
config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# config["device"] = torch.device("cpu")