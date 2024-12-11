## Intructions and Guideline to Files

`main.py`:
Launches the overall Federated Learning program

`configurations.py`:
Initialises background classes and settings

`fl_aggregation.py`:
Lists out all aggregation methods

`fl_clients.py`:
Creates a basic client class, and list out all client-related functions

`fl_strategy.py`:
Another important function to runs the Federated Learning process, called by `main.py`

### Running Code
`python3 main.py $crash_prob$ $lag$ $client_frac$ $taskname$ $taskmode$`

`crash_prob`: crash probability of clients, valid in [0,1)  
`lag`: lag tolerance for SAFA, a positive integer, e.g. 3, 5 or 10  
`client_frac`: the fraction of clients to select in each training round, valid in (0,1]  
`taskname`: federated machine learning task for demo, options are {boston, mnist, cifar10}
`taskmode`: task mode introduced by me, options are {'Semi-Async', 'FedAssets', 'FLDGNN'}

#### Examples
`python3 main.py 0.1 5 1.0 boston Semi-Async`  
`python3 main.py 0.3 5 0.3 mnist Semi-Async`  
`python3 main.py 0.0 5 0.5 cifar10 FedAssets`
`python3 main.py 0.0 5 0.3 mnist FedAssets`
`python3 main.py 0.0 5 1.0 bitcoinOTC FLDGNN`

### Default Hyperparameters
`benign_ratio`: 0.6
`poisoning_rate`: 0.7