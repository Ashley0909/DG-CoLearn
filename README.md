## Intructions and Guideline to Files

`main.py`:
Launches the overall Federated Learning program

`configurations.py`:
Initialises background classes and settings

`flgnn_dataset.py`
Includes all gnn data loading and splitting functions

`graph_partition.py`
Implements our graph partitioning algorithm

`fl_clients.py`:
Creates a basic gnn client class, and list out all client-related functions

`fl_strategy.py`:
Runs the Federated Learning process, called by `main.py`

---

### Running Code
We can run the program using:  
`python3 main.py $dataset$ $taskmode$`  

where  
`dataset`:  datasets, options are {bitcoinOTC, UCI, Brain, DBLP3, DBLP5, Reddit}  
`taskmode`: federated dynamic graph learning task mode, options are {'FLDGNN-LP', 'FLDGNN-NC'}

### Example Experiments
`python3 main.py bitcoinOTC FLDGNN-LP`  
`python3 main.py DBLP3 FLDGNN-NC`  
`python3 main.py Reddit FLDGNN-NC`  