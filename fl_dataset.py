import numpy as np
import random
import torch
from PIL import Image
from torchvision import transforms
from torchvision import datasets as torch_datasets

from fl_clients import FLClient, FLBackdoorClient
from utils import sample_dirichlet, normalize

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

class FLFedDataset:
    def __init__(self, fbd_list):
        self.fbd_list = fbd_list  # a list of FLBaseDatasets / FLLPDataset / FLNCDataset
        self.total_datasize = 0
        for fbd in self.fbd_list:
            self.total_datasize += len(fbd)  # mnist: train=60000, test=10000
        
    def __len__(self):
        return len(self.fbd_list)
    
    def __getitem__(self, item):
        return self.fbd_list[item]

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

def generate_shard_sizes(n_clients, total_size, mu, sigma):
    # Generate random sizes with mean and std deviation
    shard_sizes = np.random.randn(n_clients) * sigma + mu
    # Normalize sizes to match the total size
    return shard_sizes * total_size / shard_sizes.sum()