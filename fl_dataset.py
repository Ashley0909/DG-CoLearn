import os
import copy
import numpy as np
import random
import math
import torch
from collections import defaultdict
from PIL import Image
from torchvision import transforms
from torchvision import datasets as torch_datasets
from torch_geometric import datasets as torchgeometric_datasets
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling
from itertools import permutations

from fl_clients import FLClient, FLBackdoorClient, EdgeDevice
from utils import sample_dirichlet, normalize, process_data, localise_idx

class FLBaseDataset:
    def __init__(self, x, y, client=None):
        self.x = x  # Features
        self.y = y  # Labels
        self.length = len(y)
        self.location = client

    def __len__(self):
        return len(self.x)
    
    def bind(self, client):
        """ Bind the dataset to a client """
        assert isinstance(client, FLClient) or isinstance(client, FLBackdoorClient)
        self.location = client

class FLLPDataset:
    def __init__(self, x, edge_index, edge_label_index, edge_label, previous_edge_index, client=None):
        self.x = x
        self.edge_index = edge_index  # edge_indces of pos edges
        self.edge_label_index = edge_label_index # edge indices of pos and neg edges
        self.y = edge_label  # 1/0 labels of combined edges (pos and neg)
        self.previous_edge_index = previous_edge_index
        self.length = x.shape[0]
        self.eilength = edge_index.shape[1]
        self.ellength = edge_label.shape[0]
        self.location = client

    def __len__(self):
        return len(self.x)

    def bind(self, client):
        """ Bind the dataset to a client """
        assert isinstance(client, EdgeDevice)
        self.location = client

class FLNCDataset:
    def __init__(self, x, subnodes, edge_index, previous_edge_index, y, client=None):
        self.x = x
        self.subnodes = subnodes
        self.edge_index = edge_index
        self.previous_edge_index = previous_edge_index
        self.y = y
        self.location = client

    def __len__(self):
        return len(self.x)
    
    def bind(self, client):
        """ Bind the dataset to a client """
        assert isinstance(client, EdgeDevice)
        self.location = client

class FLFedDataset:
    def __init__(self, fbd_list):
        self.fbd_list = fbd_list  # a list of FLBaseDatasets / FLLPDataset 
        self.total_datasize = 0
        for fbd in self.fbd_list:
            self.total_datasize += len(fbd)  # mnist: train=60000, test=10000
        
    def __len__(self):
        return len(self.fbd_list)
    
    def __getitem__(self, item):
        return self.fbd_list[item]
    
class FLDataLoader:
    def __init__(self, fed_dataset, client2idx, batch_size, shuffle=False):
        self.fed_dataset = fed_dataset
        self.baseDataset_pointer = None
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_pointer = -1  # To split data into batches

        if self.shuffle:
            for ds in self.idx2data:
                ds_size = len(ds)
                rand_idcs = torch.randperm(ds_size).tolist()
                ds.data = ds.data[rand_idcs]
                ds.targets = ds.targets[rand_idcs]
        
    def __iter__(self):
        self.batch_pointer = -1
        self.baseDataset_idx = 0
        self.baseDataset_pointer = self.fed_dataset[self.baseDataset_idx]
        self.client_idx = self.baseDataset_pointer.location
        return self
    
    def __next__(self):
        self.batch_pointer += 1
        if self.batch_pointer * self.batch_size >= self.baseDataset_pointer.length:  # if no more batch for the current client
            self.batch_pointer = 0  # reset
            self.baseDataset_idx += 1  # next BaseDataset
            if self.baseDataset_idx >= len(self.fed_dataset):  # no more client to iterate through
                self.stop()
            self.baseDataset_pointer = self.fed_dataset[self.baseDataset_idx]

        right_bound = self.baseDataset_pointer.length
        this_batch_x = self.baseDataset_pointer.x[self.batch_pointer * self.batch_size:min(right_bound, (self.batch_pointer + 1) * self.batch_size)]
        this_batch_y = self.baseDataset_pointer.y[self.batch_pointer * self.batch_size:min(right_bound, (self.batch_pointer + 1) * self.batch_size)]
        location = self.baseDataset_pointer.location

        return this_batch_x, this_batch_y, location

    def stop(self):
        raise StopIteration
    
class FLGNNDataLoader:
    def __init__(self, fed_dataset, client2idx, batch_size, shuffle=False):
        self.fed_dataset = fed_dataset
        self.baseDataset_pointer = None
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_pointer = -1  # To split data into batches

        if self.shuffle:
            for ds in self.idx2data:
                ds_size = len(ds)
                rand_idcs = torch.randperm(ds_size).tolist()
                ds.data = ds.data[rand_idcs]
                ds.targets = ds.targets[rand_idcs]
        
    def __iter__(self):
        self.batch_pointer = -1
        self.baseDataset_idx = 0
        self.baseDataset_pointer = self.fed_dataset[self.baseDataset_idx]
        self.client_idx = self.baseDataset_pointer.location
        return self
    
    def __next__(self):
        self.batch_pointer += 1
        if self.batch_pointer * self.batch_size >= self.baseDataset_pointer.eilength:  # if no more batch for the current client
            self.batch_pointer = 0  # reset
            self.baseDataset_idx += 1  # next BaseDataset
            if self.baseDataset_idx >= len(self.fed_dataset):  # no more client to iterate through
                self.stop()
            self.baseDataset_pointer = self.fed_dataset[self.baseDataset_idx]

        right_bound_y = self.baseDataset_pointer.ellength
        right_bound_ei = self.baseDataset_pointer.eilength
        this_batch_x = self.baseDataset_pointer.x  # Don't need to slice, since each client has all the nodes
        this_batch_y = self.baseDataset_pointer.y[self.batch_pointer * self.batch_size:min(right_bound_y, (self.batch_pointer + 1) * self.batch_size)]
        # Changed to [:, x:xx] because we are slicing the second dimension
        this_batch_edge_index = self.baseDataset_pointer.edge_index[:, self.batch_pointer * self.batch_size : min(right_bound_ei, (self.batch_pointer + 1) * self.batch_size)]
        this_batch_edge_label_index = self.baseDataset_pointer.edge_label_index[:, self.batch_pointer * self.batch_size : min(right_bound_y, (self.batch_pointer + 1) * self.batch_size)]
        prev_edge_index = self.baseDataset_pointer.previous_edge_index

        location = self.baseDataset_pointer.location

        return this_batch_x, this_batch_edge_index, this_batch_edge_label_index, this_batch_y, prev_edge_index, location

    def stop(self):
        raise StopIteration

def load_data(task_cfg, env_cfg):
    if task_cfg.dataset == 'boston':
        data = np.loadtxt(task_cfg.path, delimiter=',', skiprows=1)
        data = normalize(data)
        data_merged = True
    elif task_cfg.dataset == 'mnist':
        # ref: https://github.com/pytorch/examples/blob/master/mnist/main.py
        mnist_train = torch_datasets.MNIST('data/mnist/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        mnist_test = torch_datasets.MNIST('data/mnist/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        train_x = mnist_train.data.view(-1, 1, 28, 28).float()
        train_y = mnist_train.targets.long()
        test_x = mnist_test.data.view(-1, 1, 28, 28).float()
        test_y = mnist_test.targets.long()

        train_data_size, test_data_size = len(train_x), len(test_x)
        data_size = train_data_size + test_data_size
        data_merged = False
    elif task_cfg.dataset == 'cifar10':
        cifar10_train = torch_datasets.CIFAR10('data/cifar10/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        cifar10_test = torch_datasets.CIFAR10('data/cifar10/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        train_x = torch.tensor(cifar10_train.data).permute(0,3,1,2).view(-1,3,32,32).float()
        train_y = torch.tensor(cifar10_train.targets).long()
        test_x = torch.tensor(cifar10_test.data).permute(0,3,1,2).view(-1, 3, 32, 32).float()
        test_y = torch.tensor(cifar10_test.targets).long()

        train_data_size, test_data_size = len(train_x), len(test_x)
        data_size = train_data_size + test_data_size
        data_merged = False
    elif task_cfg.dataset == 'cifar100':
        cifar100_train = torch_datasets.CIFAR100('data/cifar100/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        cifar100_test = torch_datasets.CIFAR100('data/cifar100/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        train_x = torch.tensor(cifar100_train.data).permute(0,3,1,2).float()
        train_y = torch.tensor(cifar100_train.targets).long()
        test_x = torch.tensor(cifar100_test.data).permute(0,3,1,2).float()
        test_y = torch.tensor(cifar100_test.targets).long()

        train_data_size, test_data_size = len(train_x), len(test_x)
        data_size = train_data_size + test_data_size
        data_merged = False
    else:
        print('E> Invalid dataset specified. Options are {boston, mnist, cifar10, cifar100}')
        exit(-1)
    
    # Partition Data
    if data_merged:
        data_size = len(data)
        train_data_size = int(data_size * env_cfg.train_frac)
        test_data_size = data_size - train_data_size
        data = torch.tensor(data).float()
        train_x = data[0:train_data_size, 0:task_cfg.in_dim]  # training data, x
        train_y = data[0:train_data_size, task_cfg.out_dim * -1:].reshape(-1, task_cfg.out_dim)  # training data, y
        test_x = data[train_data_size:, 0:task_cfg.in_dim]  # test data following, x
        test_y = data[train_data_size:, task_cfg.out_dim * -1:].reshape(-1, task_cfg.out_dim)  # test data, x
    
    return train_x, train_y, test_x, test_y, data_size

def load_lp_data(task_cfg):
    if task_cfg.dataset == 'bitcoinOTC':
        data = torchgeometric_datasets.BitcoinOTC(task_cfg.path)
    elif task_cfg.dataset == 'UCI':
        # path = download_url('http://snap.stanford.edu/data/CollegeMsg.txt.gz', task_cfg.path) # Download data if needed
        # extract_gz(path, task_cfg.path)
        # os.unlink(path)
        txt_path = os.path.join(task_cfg.path, "CollegeMsg.txt")
        data = process_data(txt_path)
    else:
        print('E> Invalid link prediction dataset specified. Options are {bitcoinOTC, UCI}')
        exit(-1)

    num_snapshots = len(data)
    hidden_conv1, hidden_conv2 = 64, 32
    last_embeddings = [torch.Tensor([[0 for _ in range(hidden_conv1)] for _ in range(data[0].num_nodes)]),torch.Tensor([[0 for _ in range(hidden_conv2)] for _ in range(data[0].num_nodes)])]
    weights = torch.zeros(data[0].num_nodes)
    task_cfg.in_dim = data[0].num_node_features
    task_cfg.out_dim = data[0].num_nodes  # number of nodes is not the output dimension, I just used out_dim to store num_nodes for init_global_model

    """ Split each snapshot into train, val and test """
    train_list, val_list, test_list, data_size = partition_lp_data("test-temporal", num_snapshots, data)

    return num_snapshots, train_list, val_list, test_list, data_size, {'last_embeddings': last_embeddings, 'weights': weights, 'num_nodes': data[0].num_nodes}

def load_nc_data(task_cfg):
    data = np.load('./data/{}.npz'.format(task_cfg.dataset))
    adjs = data['adjs']
    feature = data['attmats']
    label = data['labels']
    assert adjs.shape[1] == adjs.shape[2] == feature.shape[0] == label.shape[0]
    assert adjs.shape[0] == feature.shape[1]
    num_timesteps = adjs.shape[0]
    num_nodes = feature.shape[0]
    print("total number of nodes", num_nodes)
    # last_embeddings = [torch.Tensor([[0 for _ in range(hidden_conv1)] for _ in range(num_nodes)]), torch.Tensor([[0 for _ in range(hidden_conv2)] for _ in range(num_nodes)])]
    weights = torch.zeros(num_nodes)
    task_cfg.in_dim = feature.shape[2]
    task_cfg.out_dim = num_nodes  # number of nodes is not the output dimension, I just used out_dim to store num_nodes for init_global_model

    for node in range(num_nodes):
        adjs[:, node, node] = 0
    adjs = [adjs[t, :, :] for t in range(adjs.shape[0])]
    feature = [feature[:, t, :] for t in range(feature.shape[1])]
    label = np.argmax(label, axis=1)
    task_cfg.num_classes = max(label) + 1
    adjs = [torch.tensor(adj, dtype=torch.long).to_sparse() for adj in adjs]
    indices = [adj.indices() for adj in adjs]
    graph_snapshots = [Data(edge_index=index, num_nodes=num_nodes) for index in indices]
    feature = [torch.tensor(feat, dtype=torch.float) for feat in feature]
    label = torch.tensor(label, dtype=torch.long)
    for graph, feat in zip(graph_snapshots, feature):
        graph.x = feat
        graph.y = label

    return num_timesteps, graph_snapshots, {'last_embeddings': None, 'weights': weights, 'num_nodes': num_nodes, 'y': label}

def partition_lp_data(mode, num_snapshots, data):
    """ Partition data in train, val and test according to different scenarios:
        real-life: clients do not have t+1 data to validation (train and val both in time t)
        test-temporal: clients test the models on generalising future performance (val and test both in time t+1)
        ideal: clients train using t, val using t+1 and test using t+2

        Split edges in each snapshot
    """
    all_modes = {"real-life", "test-temporal", "ideal"}
    train_list, val_list, test_list, data_size, num_previous_edges = [], [], [], [], [0]

    for i in range(num_snapshots-1):
        # Create train and test data for each timestamp
        current = copy.deepcopy(data[i])
        current.x = torch.Tensor([[1] for _ in range(current.num_nodes)])
        num_current_edges = len(current.edge_index[0])
        num_previous_edges.append(num_previous_edges[-1] + num_current_edges)

        if mode == "real-life" or (mode not in all_modes):
            if mode not in all_modes:
                print("Invalid Mode, Split according to Real-Life mode")
            transform = RandomLinkSplit(num_val=0.25,num_test=0.0)  # RandomLinkSplit also sample/generates negative edges (same ratio as positive edges)
            train_data, val_data, _ = transform(current)
            """ CAUTION: The edge index is not modified, only edge_label = edge_label_index are split """
        
            test_data = copy.deepcopy(data[i+1])
            test_data.x = torch.Tensor([[1] for _ in range(test_data.num_nodes)])
            future_neg_edge_index = negative_sampling(  # Generate fake edges (negative edges) where A and B actually don't have a link between them
                edge_index=test_data.edge_index, # Extract positive edges (Existing Links)
                num_nodes=test_data.num_nodes,
                num_neg_samples=test_data.edge_index.size(1)) # Set the number of neg links to be equal to number of pos links (to balance the dataset)
            num_pos_edge = test_data.edge_index.size(1)
            test_data.edge_label = torch.Tensor(np.array([1 for _ in range(num_pos_edge)] + [0 for _ in range(num_pos_edge)])) # Label 1 to pos edges (a trust relationship exists), 0 to neg (no trust relationship)
            test_data.edge_label_index = torch.cat([test_data.edge_index, future_neg_edge_index], dim=-1) # Train the model with both positive and negative edges
        
        elif mode == "test-temporal":
            transform = RandomLinkSplit(num_val=0.0, num_test=0.0)  # No validation/test split in training data
            train_data, _, _ = transform(current)

            future = copy.deepcopy(data[i+1])
            future.x = torch.Tensor([[1] for _ in range(future.num_nodes)])
            transform = RandomLinkSplit(num_val=0.0, num_test=0.5) # Since the train data here is actually the validation data
            val_data, _ , test_data = transform(future)
        
        train_list.append(train_data)
        val_list.append(val_data)
        test_list.append(test_data)
        data_size.append(len(train_data.edge_label) + len(val_data.edge_label) + len(test_data.edge_label)) # Set the data size as the size of edge label (pos and neg edges)
        
    if mode == "ideal":
        for i in range(num_snapshots - 2):
            current = copy.deepcopy(data[i])
            current.x = torch.Tensor([[1] for _ in range(current.num_nodes)])
            num_current_edges = len(current.edge_index[0])
            num_previous_edges.append(num_previous_edges[-1] + num_current_edges)

            future = copy.deepcopy(data[i+1])
            future.x = torch.Tensor([[1] for _ in range(future.num_nodes)])

            t_plus_2 = copy.deepcopy(data[i+2])
            t_plus_2.x = torch.Tensor([[1] for _ in range(future.num_nodes)])

            transform = RandomLinkSplit(num_val=0.0, num_test=0.0)  # No validation/test split in training data
            train_data, _, _ = transform(current)

            transform = RandomLinkSplit(num_val=0.0, num_test=0.0)  # All for validation in time t+1
            val_data, _, _ = transform(future)

            transform = RandomLinkSplit(num_val=0.0, num_test=0.0)  # All for test in time t+2
            test_data, _, _ = transform(t_plus_2)

            train_list.append(train_data)
            val_list.append(val_data)
            test_list.append(test_data)
            data_size.append(len(train_data.edge_label) + len(val_data.edge_label) + len(test_data.edge_label)) # Set the data size as the size of edge label (pos and neg edges)
    
    return train_list, val_list, test_list, data_size

def poison_data(task_cfg, x, y, poisoning_rate=0.7, target_label=9):
    if task_cfg.dataset == 'cifar10' or task_cfg.dataset == 'cifar100':
        t_img = Image.open("./triggers/trigger_white.png").convert('RGB')
    elif task_cfg.dataset == 'mnist':
        t_img = Image.open("./triggers/trigger_white.png").convert('L')
    t_img = t_img.resize((5,5))
    transform = transforms.ToTensor()
    trigger_img = transform(t_img)

    poison_idx = random.sample(list(range(len(y))), int(len(y) * poisoning_rate))
    pimages = x[poison_idx]
    plabels = y[poison_idx]
    pimages[:,:, -5:, -5:] = trigger_img
    plabels[:] = target_label
    x = torch.cat([x, pimages], dim=0)
    y = torch.cat([y, plabels], dim=0)

    return x,y

def get_clientdata(train_x, train_y, test_x, test_y, env_cfg, task_cfg, clients, from_file=None):
    """ Build federated datasets for each client """
    train_size = len(train_x)
    test_size = len(test_x)
    data_size = train_size + test_size

    client_train_data = []
    client_test_data = []
    client_shards_sizes = []

    if env_cfg.data_dist[0] == 'E':  # Case 1: Equal-Sized Partition
        split_points_train = [0]  # partition should start from 0
        split_points_test = [0]

        eq_size_train = int(train_size / env_cfg.n_clients)
        eq_size_test = int(test_size / env_cfg.n_clients)
        for i in range(env_cfg.n_clients):
            split_points_train.append((i+1) * eq_size_train)
            split_points_test.append((i+1) * eq_size_test)
            client_shards_sizes.append(eq_size_train + eq_size_test)
    elif env_cfg.data_dist[0] == 'X':  # Case 2: Exponential distribution
        rerand = True
        while rerand: # In case of illegal local data size
            rerand = False
            client_shards_sizes = []
            # uniform split points, in percentage
            split_points_pct = np.append([0, 1], np.random.random_sample(size=env_cfg.n_clients-1))
            split_points_pct.sort()
            split_points_train = (split_points_pct * train_size).astype(int)
            split_points_test = (split_points_pct * test_size).astype(int)
            # validity check
            for i in range(env_cfg.n_clients):
                quota = split_points_train[i+1] - split_points_train[i] + split_points_test[i+1] - split_points_test[i]
                if quota < max(20, env_cfg.batch_size):  # check each shard size
                    rerand = True  # can't be too small
                    break
                else:
                    client_shards_sizes.append(quota)
    elif env_cfg.data_dist[0] == 'N':  # Case 3: Normal distribution
        mu = data_size / env_cfg.n_clients
        sigma = env_cfg.data_dist[1] * mu

        rerand = True
        while rerand:
            # directly generate sizes of shards, temporarily
            client_shards_sizes = np.random.randn(env_cfg.n_clients) * sigma + mu
            rerand = False
            # make it add up to data_size
            client_shards_sizes = client_shards_sizes * data_size / client_shards_sizes.sum()
            # validity check
            for s in client_shards_sizes:
                if s < max(20, env_cfg.batch_size):
                    rerand = True
                    break
        # now compose train and test partitions separately
        split_points_train = [0]
        last_point_train = 0
        split_points_test = [0]
        last_point_test = 0
        for s in client_shards_sizes:
            # for training
            split_points_train.append(last_point_train + int(s * env_cfg.train_frac))
            last_point_train += int(s * env_cfg.train_frac)
            # for test
            split_points_test.append(last_point_test + int(s * env_cfg.test_frac))
            last_point_test += int(s * env_cfg.test_frac)

        # round up to pre-determined sizes
        split_points_train[-1] = train_size
        split_points_test[-1] = test_size
        # recalibrate client data shards
        for i in range(env_cfg.n_clients):
            quota = split_points_train[i+1] - split_points_train[i] + split_points_test[i+1] - split_points_test[i]
            client_shards_sizes[i] = quota
        client_shards_sizes = client_shards_sizes.astype(int)
    elif env_cfg.data_dist[0] == 'D': # Case 4: Dirichlet distribution
        alpha = env_cfg.data_dist[1]
        split_points_train = [[] for _ in range(env_cfg.n_clients)]
        split_points_test = [[] for _ in range(env_cfg.n_clients)]

        split_points_train, client_shards_sizes = sample_dirichlet(split_points_train, train_y, env_cfg, alpha, client_shards_sizes)
        split_points_test, client_shards_sizes = sample_dirichlet(split_points_test, test_y, env_cfg, alpha, client_shards_sizes)
    else:
        print('Error> Invalid data distribution option. Options are {E, X, (N, rlt_sgm)}')
        exit(0)

    if from_file:
        split_points_train = (np.loadtxt(from_file) * train_size).astype(int)
        split_points_test = (np.loadtxt(from_file) * test_size).astype(int)
        client_shards_sizes[0] = split_points_train[0] + split_points_test[0]
        for k in range(1, env_cfg.n_clients):
            train_shards = split_points_train[k] - split_points_train[k-1]
            test_shards = split_points_train[k] - split_points_train[k-1]
            client_shards_sizes.append(train_shards+test_shards)

    # split data and dispatch
    num_benign = int(env_cfg.n_clients * env_cfg.benign_ratio)
    for i in range(env_cfg.n_clients):
        if env_cfg.data_dist[0] == 'D':
            xtrain = train_x[split_points_train[i]]
            ytrain = train_y[split_points_train[i]]
            xtest = test_x[split_points_test[i]]
            ytest = test_y[split_points_test[i]]
        else:
            xtrain = train_x[split_points_train[i]: split_points_train[i+1]]
            ytrain = train_y[split_points_train[i]: split_points_train[i+1]]
            xtest = test_x[split_points_test[i]: split_points_test[i+1]]
            ytest = test_y[split_points_test[i]: split_points_test[i+1]]
        if env_cfg.mode == 'FedAssets' and i >= num_benign:
            poisoned_trainx, poisoned_trainy = poison_data(task_cfg, xtrain, ytrain)
            poisoned_testx, poisoned_testy = poison_data(task_cfg, xtest, ytest)
            client_train_data.append(FLBaseDataset(poisoned_trainx, poisoned_trainy))
            client_test_data.append(FLBaseDataset(poisoned_testx, poisoned_testy))
        else:
            client_train_data.append(FLBaseDataset(xtrain, ytrain))
            client_test_data.append(FLBaseDataset(xtest, ytest))
        
        # allocate the BaseDataset to clients
        client_train_data[i].bind(clients[i])
        client_test_data[i].bind(clients[i])

    fed_data_train = FLFedDataset(client_train_data)
    fed_data_test = FLFedDataset(client_test_data)

    return fed_data_train, fed_data_test, client_shards_sizes

def allocate_clientsubnodes(env_cfg, arg, clients):
    """ Distribute Subnodes to Clients """
    subnodes_split, subnodes_list = [0], []
    num_nodes = arg['num_nodes']

    if env_cfg.mode == "FLDGNN-LP":
        # Shuffle node indices for subnode allocation
        rand_node_idc = torch.randperm(num_nodes).tolist()

        if env_cfg.data_dist[0] == 'E': # Case 1: Equal Distribution in terms of Number of Nodes
            subnode_eq = int(num_nodes / env_cfg.n_clients)
            for i in range(env_cfg.n_clients):
                subnodes_split.append((i+1) * subnode_eq)
        elif env_cfg.data_dist[0] == 'N': # Case 2: Normal Distribution in terms of Number of Nodes
            mu= num_nodes / env_cfg.n_clients
            sigma = env_cfg.data_dist[1] * mu
            node_shardsizes = generate_shard_sizes(env_cfg.n_clients, num_nodes, mu, sigma)

            last_point = 0
            for split in node_shardsizes:
                subnodes_split.append(last_point + int(split))
                last_point += int(split)

            subnodes_split[-1] = num_nodes

        for i in range(env_cfg.n_clients):
            client_subnodes = rand_node_idc[subnodes_split[i]: subnodes_split[i+1]]
            client_subnodes.sort()
            subnodes_list.append(client_subnodes)

    elif env_cfg.mode == "FLDGNN-NC":
        if env_cfg.data_dist[0] == 'E': # Case 1: Equal Distribution in terms of Number of Nodes
            rand_node_idc = torch.randperm(num_nodes).tolist()
            subnode_eq = int(num_nodes / env_cfg.n_clients)
            for i in range(env_cfg.n_clients):
                subnodes_split.append((i+1) * subnode_eq)

            for i in range(env_cfg.n_clients):
                client_subnodes = rand_node_idc[subnodes_split[i]: subnodes_split[i+1]]
                client_subnodes.sort()
                subnodes_list.append(client_subnodes)
        elif env_cfg.data_dist[0] == 'Label': # Case 2: Label-skew Distribution, nodes are divided by labels
            y = arg['y']
            unique_labels = torch.unique(y)
            node_by_label = defaultdict(list)
            node_indices = np.arange(num_nodes)
            for node in node_indices:
                node_by_label[y[node].item()].append(node)

            aval_perm = list(permutations(unique_labels.numpy(), 2)) # output all the possible permutation of 2 given the unique labels

            all_assigned_nodes = set()
            subnodes_list = []

            for i in range(env_cfg.n_clients):
                idx = random.randrange(len(aval_perm)) # pick one permutation as the majority label pair
                majority_labels = aval_perm.pop(idx)
                pick_nodes = [] # all the available nodes for the client to choose from
                for label in majority_labels:
                    pick_nodes += node_by_label[label]

                pick_nodes = list(set(pick_nodes) - all_assigned_nodes)
                majority_sample = math.floor(len(pick_nodes) * 0.8) # Client takes 80% of the majoirty data to form its dataset
                other_samples = math.ceil(len(pick_nodes) * 0.2)    # For the remaining client data, we take from other labels
                client_nodes = random.sample(pick_nodes, majority_sample)
                all_assigned_nodes.update(client_nodes)
                other_nodes = random.sample(np.setdiff1d(node_indices, list(all_assigned_nodes)).tolist(), other_samples)
                all_assigned_nodes.update(other_nodes)
                client_nodes = client_nodes + other_nodes
                subnodes_list.append(client_nodes)
                print(f'Client {i} has {len(client_nodes)} nodes')
                
                # Set previous embedding for each client (In NC, prev embedding shape varies among clients)
                hidden_conv1, hidden_conv2 = 64, 32
                clients[i].prev_ne = [torch.Tensor([[0 for _ in range(hidden_conv1)] for _ in range(len(client_nodes))]), torch.Tensor([[0 for _ in range(hidden_conv2)] for _ in range(len(client_nodes))])] # Set the PE shape of each client to be [#subnodes, in_dim] respectively

    return subnodes_list

def get_random_clientdata(train, val, test, env_cfg, clients):
    """ Build GNN federated datasets for each client by randomly allocating edges """
    if env_cfg.mode == "FLDGNN-LP":
        n_clients = env_cfg.n_clients
        client_train_data, client_val_data, client_test_data = [], [], [] # list of FLGNNDatasets
        client_shards_sizes = [] # number of pos and neg edges in train + val + test

        # train and validation have the same Edge_Index size
        ei_train_size, ei_val_size, ei_test_size = train.edge_index.shape[1], val.edge_index.shape[1], test.edge_index.shape[1]
        ei_train_split, ei_val_split, ei_test_split = [0], [0], [0]

        # train, validation and test have different Edge_Label size (= Edge_Label_Index size)
        el_train_size, el_val_size, el_test_size = train.edge_label.shape[0], val.edge_label.shape[0], test.edge_label.shape[0]
        el_train_split, el_val_split, el_test_split = [0], [0], [0]

        if env_cfg.shuffle: # Shuffle the data before splitting if necessary (Definitely have to, or else the first few snapshots will not have negative edges, vice versa for the last few snapshots)
            # No need to shuffle edge_index and x, just need to shuffle edge_label and edge_label_index (should keep their correspondance as well)
            train_rand_idc = torch.randperm(el_train_size).tolist()
            val_rand_idc = torch.randperm(el_val_size).tolist()
            test_rand_idc = torch.randperm(el_test_size).tolist()

            train.edge_label = train.edge_label[train_rand_idc]
            val.edge_label = val.edge_label[val_rand_idc]
            test.edge_label = test.edge_label[test_rand_idc]
            train.edge_label_index = train.edge_label_index[:, train_rand_idc]
            val.edge_label_index = val.edge_label_index[:, val_rand_idc]
            test.edge_label_index = test.edge_label_index[:, test_rand_idc]

        if env_cfg.data_dist[0] == 'E': # Case 1: Equal Distribution
            ei_eq_train, ei_eq_val, ei_eq_test = int(ei_train_size / n_clients), int(ei_val_size / n_clients), int(ei_test_size / n_clients)
            el_eq_train, el_eq_val, el_eq_test = int(el_train_size / n_clients), int(el_val_size / n_clients), int(el_test_size / n_clients)
            for i in range(env_cfg.n_clients):
                ei_train_split.append((i+1) * ei_eq_train)
                ei_val_split.append((i+1) * ei_eq_val)
                ei_test_split.append((i+1) * ei_eq_test)
                el_train_split.append((i+1) * el_eq_train)
                el_val_split.append((i+1) * el_eq_val)
                el_test_split.append((i+1) * el_eq_test)

                client_shards_sizes.append(el_eq_train + el_eq_val + el_eq_test) # shard size is to record the number of pos and neg edges => edge_label size
        elif env_cfg.data_dist[0] == 'N': # Case 2: Normal Distribution
            # Consider Edge_Index
            mu_train, mu_val, mu_test = ei_train_size/n_clients, ei_val_size/n_clients, ei_test_size/n_clients
            sigma_train, sigma_val, sigma_test = env_cfg.data_dist[1]*mu_train, env_cfg.data_dist[1]*mu_val, env_cfg.data_dist[1]*mu_test

            train_shardsizes = generate_shard_sizes(n_clients, ei_train_size, mu_train, sigma_train)
            val_shardsizes = generate_shard_sizes(n_clients, ei_val_size, mu_val, sigma_val)
            test_shardsizes = generate_shard_sizes(n_clients, ei_test_size, mu_test, sigma_test)

            # compose train and test partitions separately
            last_point_train, last_point_val, last_point_test = 0, 0, 0
            for s_train, s_val, s_test in zip(train_shardsizes, val_shardsizes, test_shardsizes):
                ei_train_split.append(last_point_train + int(s_train))
                last_point_train += int(s_train)
                ei_val_split.append(last_point_val + int(s_val))
                last_point_val += int(s_val)
                ei_test_split.append(last_point_test + int(s_test))
                last_point_test += int(s_test)

            # round up to pre-determined sizes
            ei_train_split[-1], ei_val_split[-1], ei_test_split[-1] = ei_train_size, ei_val_size, ei_test_size

            # Consider Edge_Label or Edge_Label_Index
            mu_train, mu_val, mu_test = el_train_size/n_clients, el_val_size/n_clients, el_test_size/n_clients
            sigma_train, sigma_val, sigma_test = env_cfg.data_dist[1]*mu_train, env_cfg.data_dist[1]*mu_val, env_cfg.data_dist[1]*mu_test

            train_shardsizes = generate_shard_sizes(n_clients, el_train_size, mu_train, sigma_train)
            val_shardsizes = generate_shard_sizes(n_clients, el_val_size, mu_val, sigma_val)
            test_shardsizes = generate_shard_sizes(n_clients, el_test_size, mu_test, sigma_test)

            # Use Edge_Label size to generate client_shard_sizes
            client_shards_sizes.append(train_shardsizes + val_shardsizes + test_shardsizes)

            # compose train, val and test partitions separately
            last_point_train, last_point_val, last_point_test = 0, 0, 0
            for s_train, s_val, s_test in zip(train_shardsizes, val_shardsizes, test_shardsizes):
                el_train_split.append(last_point_train + int(s_train))
                last_point_train += int(s_train)
                el_val_split.append(last_point_val + int(s_val))
                last_point_val += int(s_val)
                el_test_split.append(last_point_test + int(s_test))
                last_point_test += int(s_test)

        # Allocate Dataset Instance
        for i in range(env_cfg.n_clients):
            client_train_data.append(FLLPDataset(train.x, train.edge_index[:, ei_train_split[i]: ei_train_split[i+1]], train.edge_label_index[:, el_train_split[i]: el_train_split[i+1]],
                                                train.edge_label[el_train_split[i]: el_train_split[i+1]], clients[i].prev_edge_index))
            client_val_data.append(FLLPDataset(val.x, val.edge_index[:, ei_val_split[i]: ei_val_split[i+1]], val.edge_label_index[:, el_val_split[i]: el_val_split[i+1]],
                                                val.edge_label[el_val_split[i]: el_val_split[i+1]], clients[i].prev_edge_index))
            client_test_data.append(FLLPDataset(test.x, test.edge_index[:, ei_test_split[i]: ei_test_split[i+1]], test.edge_label_index[:, el_test_split[i]: el_test_split[i+1]],
                                                test.edge_label[el_test_split[i]: el_test_split[i+1]], clients[i].prev_edge_index))
                    
            # allocate the BaseDataset to clients
            client_train_data[i].bind(clients[i])
            client_val_data[i].bind(clients[i])
            client_test_data[i].bind(clients[i])

            # record edge index for next snapshot
            clients[i].prev_edge_index = train.edge_index[:, ei_train_split[i]: ei_train_split[i+1]]

    fed_data_train = FLFedDataset(client_train_data)
    fed_data_val = FLFedDataset(client_val_data)
    fed_data_test = FLFedDataset(client_test_data)

    return fed_data_train, fed_data_val, fed_data_test, client_shards_sizes

def get_effi_clientdata(train, val, test, env_cfg, clients, subnode_list):
    """
        A more efficient way to learn graphs through meaningful edge splitting. Given subnodes of each clients to start with, allocate relevant edges to the client for each snapshot
    """
    client_train_data, client_val_data, client_test_data = [], [], [] # list of FLLPDatasets
    # Build GNN federated datasets for each client
    client_shards_sizes = [] # number of pos and neg edges in train + val + test

    # Allocate Edges according Subnodes
    for i in range(env_cfg.n_clients):
        clients[i].subnodes = subnode_list[i]
        subnodes_tensor = torch.tensor(clients[i].subnodes)
        ei_train_idc, ei_val_idc, ei_test_idc = torch.where(torch.isin(train.edge_index[1], subnodes_tensor))[0], torch.where(torch.isin(val.edge_index[1], subnodes_tensor))[0], torch.where(torch.isin(test.edge_index[1], subnodes_tensor))[0]
        el_train_idc, el_val_idc, el_test_idc = torch.where(torch.isin(train.edge_label_index[1], subnodes_tensor))[0], torch.where(torch.isin(val.edge_label_index[1], subnodes_tensor))[0], torch.where(torch.isin(test.edge_label_index[1], subnodes_tensor))[0]
        
        client_shards_sizes.append(len(el_train_idc)+len(el_val_idc)+len(el_test_idc))

        client_train_data.append(FLLPDataset(train.x, train.edge_index[:, ei_train_idc], train.edge_label_index[:, el_train_idc], train.edge_label[el_train_idc], clients[i].prev_edge_index))
        client_val_data.append(FLLPDataset(val.x, val.edge_index[:, ei_val_idc], val.edge_label_index[:, el_val_idc], val.edge_label[el_val_idc], clients[i].prev_edge_index))
        client_test_data.append(FLLPDataset(test.x, test.edge_index[:, ei_test_idc], test.edge_label_index[:, el_test_idc], test.edge_label[el_test_idc], clients[i].prev_edge_index))
                
        # allocate the BaseDataset to clients
        client_train_data[i].bind(clients[i])
        client_val_data[i].bind(clients[i])
        client_test_data[i].bind(clients[i])

        # record edge index for next snapshot
        clients[i].prev_edge_index = train.edge_index[:, ei_train_idc]

    client_shards_sizes = np.array(client_shards_sizes)

    fed_data_train = FLFedDataset(client_train_data)
    fed_data_val = FLFedDataset(client_val_data)
    fed_data_test = FLFedDataset(client_test_data)

    return fed_data_train, fed_data_val, fed_data_test, client_shards_sizes

def get_nc_clientdata(env_cfg, t, data, clients, subnode_list, mode):
    """ 
        Partition data in train, val and test according to different scenarios:
            real-life: clients do not have t+1 data to validation (train and val both in time t)
            test-temporal: clients test the models on generalising future performance (val and test both in time t+1)
    """
    client_train_data, client_val_data, client_test_data = [], [], [] # list of FLGNNDatasets
    client_shard_sizes = []
    g_0 = data[t]
    g_1 = data[t+1]
    data_size = g_0.num_nodes
    for i in range(env_cfg.n_clients):
        subnodes = sorted(subnode_list[i])
        num_nodes = len(subnodes)
        clients[i].subnodes = subnodes
        perm = torch.tensor(subnodes) # Get the global index
        shuffled_indices = torch.randperm(perm.size(0))
        shuff_perm = perm[shuffled_indices]  # Use the permutation to shuffle perm
        # perm = torch.randperm(num_nodes) # Get the local index

        glob_to_loc = {node: i for i, node in enumerate(subnodes)} # create mapping

        # Split nodes into train, val and test
        if mode == "real-life":
            train_ratio, val_ratio = 0.75, 0.25
            train_boundary = math.floor(num_nodes * train_ratio)
            val_boundary = math.floor(num_nodes * (train_ratio + val_ratio))
            train_idx, _ = torch.sort(shuff_perm[:train_boundary])
            val_idx, _ = torch.sort(shuff_perm[train_boundary:val_boundary])

            # Localise indices
            train_edge_index, local_train_idx = localise_idx(glob_to_loc, g_0.edge_index, train_idx)
            val_edge_index, local_val_idx = localise_idx(glob_to_loc, g_0.edge_index, val_idx)
            test_edge_index, local_test_idx = localise_idx(glob_to_loc, g_1.edge_index, perm)
            print(f'Client {i} has {len(local_train_idx)} training nodes, {len(local_val_idx)} validation nodes, {num_nodes} testing nodes')

            client_train_data.append(FLNCDataset(g_0.x[train_idx], local_train_idx, train_edge_index, clients[i].prev_edge_index, g_0.y[train_idx])) # Only training need previous edges
            client_val_data.append(FLNCDataset(g_0.x[val_idx], local_val_idx, val_edge_index, None, g_0.y[val_idx]))
            client_test_data.append(FLNCDataset(g_1.x[perm], local_test_idx, test_edge_index, None, g_1.y[perm]))
 
        elif mode == 'test-temporal':
            val_ratio, test_ratio = 0.5, 0.5
            val_boundary = math.floor(num_nodes * val_ratio)
            test_boundary = math.floor(num_nodes * (val_ratio + test_ratio))
            val_idx = sorted(shuff_perm[:val_boundary])
            test_idx = sorted(shuff_perm[val_boundary:test_boundary])
            
            train_edge_index, local_train_idx = localise_idx(glob_to_loc, g_0.edge_index, perm)
            val_edge_index, local_val_idx = localise_idx(glob_to_loc, g_1.edge_index, val_idx)
            test_edge_index, local_test_idx = localise_idx(glob_to_loc, g_1.edge_index, test_idx)
            print(f'Client {i} has {len(local_train_idx)} training nodes, {len(local_val_idx)} validation nodes, {num_nodes} testing nodes')

            client_train_data.append(FLNCDataset(g_0.x[perm], local_train_idx, train_edge_index, clients[i].prev_edge_index, g_0.y[perm])) # Only training need previous edges (corrected: each client only has their subnode graph as training, not the whole graph)
            client_val_data.append(FLNCDataset(g_1.x[val_idx], local_val_idx, val_edge_index, None, g_1.y[val_idx]))
            client_test_data.append(FLNCDataset(g_1.x[test_idx], local_test_idx, test_edge_index, None, g_1.y[test_idx]))

        else:
            print(">E Invalid Split mode. Options: ['real-life', 'test-temporal']")
            exit(-1)

        clients[i].compute_weights(train_edge_index) # Compute Weights for each client
        clients[i].prev_edge_index = train_edge_index # Clients get new edge index in training by comparing with prev_edge_index

        # Allocate the BaseDataset to clients
        client_train_data[i].bind(clients[i])
        client_val_data[i].bind(clients[i])
        client_test_data[i].bind(clients[i])
        client_shard_sizes.append(num_nodes)

    fed_data_train = FLFedDataset(client_train_data)
    fed_data_val = FLFedDataset(client_val_data)
    fed_data_test = FLFedDataset(client_test_data)

    return data_size, client_shard_sizes, fed_data_train, fed_data_val, fed_data_test

def generate_shard_sizes(n_clients, total_size, mu, sigma):
    # Generate random sizes with mean and std deviation
    shard_sizes = np.random.randn(n_clients) * sigma + mu
    # Normalize sizes to match the total size
    return shard_sizes * total_size / shard_sizes.sum()