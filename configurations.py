import torch

from fl_clients import FLClient, FLBackdoorClient, EdgeDevice
from fl_models import MLmodelReg, MLmodelSVM, MLmodelCNN, MLmodelResNet, BasicBlock, ROLANDGNN

class EnvSettings:
    """
    Environment Settings for FL

    """

    def __init__(self, n_clients, n_rounds, n_epochs, batch_size, train_frac, shuffle, pick_frac, benign_ratio, data_dist, client_dist, perf_dist, crash_dist, keep_best, device, showplot, bw_set, max_T):
        self.mode = None
        self.n_clients = n_clients
        self.n_rounds = n_rounds
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.train_frac = train_frac
        self.test_frac = 1 - self.train_frac
        self.shuffle = shuffle   # shuffle the data if True
        self.pick_frac = pick_frac # fraction of clients picked for training
        self.benign_ratio = benign_ratio # fraction of benign clients
        self.data_dist = data_dist # local data distribution [(E, None): equal-sized / (N, sig): Normal, mu = total_size/n_clients, sigma = sig * mu / (X, None): Exponential] / (D, alpha): Dirichlet
        self.client_dist = client_dist # Same data distribution options, for client data splitting
        self.perf_dist = perf_dist # client performance distribution [same as above]
        self.crash_dist = crash_dist # client crash probability distri. [(E, prob): equal prob to crash, (U, (low, high)): Uniform distribution]
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
    """
    Task Settings for FL

    """

    def __init__(self, task_type, dataset, path, in_dim, out_dim, optimizer='SGD', num_classes=10, loss=None, lr=0.01, lr_decay=1.0, poisoning_rate=0.0):
        self.task_type = task_type
        self.dataset = dataset
        self.num_classes = num_classes
        self.path = path
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.optimizer = optimizer
        self.loss = loss
        self.lr = lr
        self.lr_decay = lr_decay
        self.poisoning_rate = poisoning_rate
        self.model_size = 10.0  #10MB

# A function that will trigger when an event occurred
class EventHandler:
    def __init__(self, state_names):
        """ Initialise a state """
        assert state_names is not None  # If the statement is False, terminate the program
        self.states = {sn: 0.0 for sn in state_names}

    def get_state(self, state_name):
        return self.states[state_name]
    
    def add_sequential(self, state_name, value):
        """ Add a sequential event to the system by changing a specific state"""
        self.states[state_name] += value

    def add_parallel(self, state_name, value, reduce='max'):
        """ Add a parallel event to the system using a specific reduce method 'none', 'max' or 'sum' """
        if reduce == 'none':
            self.states[state_name] += value
        elif reduce == 'max':
            self.states[state_name] += max(value)
        elif reduce == 'sum':
            self.states[state_name] += sum(value)
        else:
            print("[Error] Wrong reduce method specified. Available options: 'none', 'max', 'sum' ")

def init_config(task2run, pick_C, crash_prob, bw_set):
    if task2run == 'boston':
        ''' Boston housing regression settings (ms per epoch)'''
        # ref: https://archive.ics.uci.edu/ml/machine-learning-databases/housing/ 
        env_cfg = EnvSettings(n_clients=5, n_rounds=100, n_epochs=3, batch_size=5, train_frac=0.7, shuffle=False, pick_frac=pick_C, benign_ratio=1.0, data_dist=('N', 0.3), client_dist=('N', 0.3), perf_dist=('X', None), crash_dist=('E', crash_prob),
                              keep_best=True, dev='cpu', showplot=False, bw_set=bw_set, max_T=830)
        task_cfg = TaskSettings(task_type='Reg', dataset='boston', path='data/boston_housing.csv', in_dim=12, out_dim=1, optimizer='SGD', loss='mse', lr=1e-4, lr_decay=1.0)
    elif task2run == 'mnist':
        ''' MNIST digits classification task settings (3s per epoch on GPU)'''
        env_cfg = EnvSettings(n_clients=50, n_rounds=100, n_epochs=5, batch_size=40, train_frac=6.0/7.0, shuffle=False, pick_frac=pick_C, benign_ratio=1.0, data_dist=('E', None), client_dist=('E', None), perf_dist=('X', None), crash_dist=('E', crash_prob),
                              keep_best=True, device='gpu', showplot=False, bw_set=bw_set, max_T=5600)
        task_cfg = TaskSettings(task_type='CNN', dataset='mnist', path='data/MNIST/', in_dim=None, out_dim=None, optimizer='SGD', loss='nllLoss', lr=1e-3, lr_decay=1.0)
    elif task2run in ['cifar10', 'cifar100']:
        env_cfg = EnvSettings(n_clients=50, n_rounds=10, n_epochs=5, batch_size=20, train_frac=6.0/7.0, shuffle=False, pick_frac=pick_C, benign_ratio=0.6, data_dist=('E', None), client_dist=('E', None), perf_dist=('X', None), crash_dist=('E', crash_prob),
                              keep_best=True, device='gpu', showplot=False, bw_set=bw_set, max_T=5600)
        if task2run == 'cifar10':
            task_cfg = TaskSettings(task_type='ResNet', dataset=task2run, num_classes=10, path=f'data/{task2run}/', in_dim=None, out_dim=None, optimizer='SGD', loss='nllLoss', lr=1e-2, lr_decay=5e-4)
        else:    
            task_cfg = TaskSettings(task_type='ResNet', dataset=task2run, num_classes=100, path=f'data/{task2run}/', in_dim=None, out_dim=None, optimizer='SGD', loss='nllLoss', lr=1e-2, lr_decay=5e-4)
    elif task2run in ['bitcoinOTC', 'UCI']:
        env_cfg = EnvSettings(n_clients=2, n_rounds=2, n_epochs=30, batch_size=20, train_frac=0.5, shuffle=True, pick_frac=pick_C, benign_ratio=1.0, data_dist=('N', 0.3), client_dist=('N', 0.3), perf_dist=('X', None), crash_dist=('E', crash_prob),
                              keep_best=True, device='gpu', showplot=False, bw_set=bw_set, max_T=5600)
        task_cfg = TaskSettings(task_type='LP', dataset=task2run, path=f'data/{task2run}/', in_dim=None, out_dim=None, optimizer='Adam', loss='bce', lr=1e-3, lr_decay=5e-3)
    elif task2run in ['Brain', 'DBLP3', 'DBLP5', 'Reddit']:
        env_cfg = EnvSettings(n_clients=2, n_rounds=2, n_epochs=30, batch_size=20, train_frac=0.75, shuffle=True, pick_frac=pick_C, benign_ratio=1.0, data_dist=('Min_Cut', None), client_dist=('Label', None), perf_dist=('X', None), crash_dist=('E', crash_prob),
                              keep_best=True, device='gpu', showplot=False, bw_set=bw_set, max_T=5600)
        task_cfg = TaskSettings(task_type='NC', dataset=task2run, path=f'data/{task2run}/', in_dim=None, out_dim=None, optimizer='Adam', loss='ce', lr=1e-3, lr_decay=5e-3)
    else:
        print('[Err] Invalid task name provided. Options are {boston, mnist, cifar10, cifar100, bitcoinOTC, UCI, Brain, DBLP3, DBLP5, Reddit}')
        exit(0)

    return env_cfg, task_cfg

def init_FL_clients(num_clients):
    clients = []
    cm_map = {}  # client-model mapping
    for i in range(num_clients):
        clients.append(FLClient(id='client_' + str(i)))
        cm_map['client_' + str(i)] = i  # Client i with Model i
    return clients, cm_map

def init_FLBackdoor_clients(num_clients, br):
    clients = []
    cm_map = {}
    mali_map = {}
    num_benign = int(num_clients * br)
    for i in range(num_clients):
        if i < num_benign:
            clients.append(FLBackdoorClient(id='client_' + str(i), malicious=0))
            cm_map['client_' + str(i)] = i
            mali_map[i] = 0
        else:
            clients.append(FLBackdoorClient(id='client_' + str(i), malicious=1))
            cm_map['client_' + str(i)] = i
            mali_map[i] = 1
    
    return clients, cm_map, mali_map

def init_GNN_clients(num_clients, last_ne, weights):
    """ last_ne is the same for each client, with shape [total_num_nodes, 64] and [total_num_nodes, 32] """
    clients = []
    cm_map = {}
    for i in range(num_clients):
        clients.append(EdgeDevice(id=f'client_{i}', prev_ne=last_ne, weights=weights, subnodes=None))
        cm_map[f'client_{i}'] = i

    return clients, cm_map

def init_global_model(env_cfg, task_cfg):
    model = None
    device = env_cfg.device

    if device.type == 'cuda':
        torch.set_default_dtype(torch.float32)

    if task_cfg.task_type == 'Reg':
        model = MLmodelReg(in_features=task_cfg.in_dim, out_features=task_cfg.out_dim).to(device)
    elif task_cfg.task_type == 'SVM':
        model = MLmodelSVM(in_features=task_cfg.in_dim).to(device)
    elif task_cfg.task_type == 'CNN':
        model = MLmodelCNN(classes=10).to(device)
    elif task_cfg.task_type == 'ResNet':
        model = MLmodelResNet(BasicBlock, [2, 2, 2, 2], num_classes=task_cfg.num_classes).to(device)
    elif task_cfg.task_type == 'LP':
        model = ROLANDGNN(device=device, input_dim=task_cfg.in_dim, output_dim=2, num_nodes=task_cfg.out_dim, update='gru').to(device)
        model.reset_parameters()
    elif task_cfg.task_type == 'NC':
        model = ROLANDGNN(device=device, input_dim=task_cfg.in_dim, output_dim=task_cfg.num_classes, num_nodes=task_cfg.out_dim, update='gru').to(device)
        model.reset_parameters()
    torch.set_default_dtype(torch.float32)
    return model