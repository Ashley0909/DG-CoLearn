import torch

from fl_clients import EdgeDevice
from fl_models import ROLANDGNN

from gnn_recurrent import GNN

class EnvSettings:
    """ Environment Settings for FL """

    def __init__(self, n_clients, n_rounds, n_epochs, keep_best=True, device='gpu', showplot=False, bw_set=(0.175, 1250), max_T=830):
        self.mode = None
        self.n_clients = n_clients
        self.n_rounds = n_rounds
        self.n_epochs = n_epochs
        self.keep_best = keep_best # keep best global model if True
        self.device = device
        self.showplot = showplot
        self.bw_set = bw_set  # bandwidth setting = (bw_d, bw_u, bw_server)
        self.max_T = max_T  # max time

        if device == 'cpu':
            self.device = torch.device('cpu')
        elif device == 'gpu' and torch.cuda.is_available():
            self.device = torch.device('cuda:0')

class TaskSettings:
    """ Task Settings for FL """

    def __init__(self, task_type, dataset, path, in_dim, out_dim, batch_size=5, optimizer='SGD', num_classes=10, loss=None, lr=0.01, lr_decay=1.0, poisoning_rate=0.0):
        self.task_type = task_type
        self.dataset = dataset
        self.num_classes = num_classes
        self.path = path
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss = loss
        self.lr = lr
        self.lr_decay = lr_decay
        self.poisoning_rate = poisoning_rate
        self.model_size = 10.0  #10MB

def init_config(dataset, bw_set, round, epoch):
    if dataset == 'SBM': # Generate graphs
        #10,10
        #30,10
        #50,10
        env_cfg = EnvSettings(n_clients=10, n_rounds=round, n_epochs=epoch, keep_best=True, device='gpu', bw_set=bw_set, max_T=5600)
        task_cfg = TaskSettings(task_type='NC', dataset=dataset, path=f'data/{dataset}/', in_dim=None, out_dim=None, batch_size=5, optimizer='Adam', loss='ce', lr=0.04, lr_decay=1e-1)
    elif dataset in ['bitcoinOTC', 'UCI']:
        env_cfg = EnvSettings(n_clients=10, n_rounds=round, n_epochs=epoch, keep_best=True, device='gpu', bw_set=bw_set, max_T=5600)
        task_cfg = TaskSettings(task_type='LP', dataset=dataset, path=f'data/{dataset}/', in_dim=None, out_dim=None, batch_size=5, optimizer='Adam', loss='ce', lr=0.03, lr_decay=0.1)
    elif dataset in ['DBLP3', 'DBLP5', 'Reddit']:
        env_cfg = EnvSettings(n_clients=10, n_rounds=round, n_epochs=epoch, keep_best=True, device='gpu', bw_set=bw_set, max_T=5600)
        task_cfg = TaskSettings(task_type='NC', dataset=dataset, path=f'data/{dataset}/', in_dim=None, out_dim=None, batch_size=5, optimizer='Adam', loss='ce', lr=0.04, lr_decay=1e-1)
    else:
        print('[Err] Invalid dataset provided. Options are {SBM, bitcoinOTC, UCI, DBLP3, DBLP5, Reddit}')
        exit(0)

    return env_cfg, task_cfg

def init_GNN_clients(num_clients, last_ne):
    """ last_ne is the same for each client, with shape [total_num_nodes, hidden_conv_1] and [total_num_nodes, hidden_conv_2] """
    clients = []
    cm_map = {}
    for i in range(num_clients):
        clients.append(EdgeDevice(id=f'client_{i}', prev_ne=last_ne, subnodes=None))
        cm_map[f'client_{i}'] = i

    return clients, cm_map

def init_global_model(env_cfg, task_cfg, arg):
    model = None
    device = env_cfg.device

    if device.type == 'cuda':
        torch.set_default_dtype(torch.float32)

    model = GNN(dim_in=task_cfg.in_dim, dim_out=task_cfg.out_dim, glob_shape=arg['num_nodes'], task_type=task_cfg.task_type)
    torch.set_default_dtype(torch.float32)
    return model