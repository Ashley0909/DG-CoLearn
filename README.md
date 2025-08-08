# To use the most updated implementation, switch to `fast_gpa` branch.

# Instructions and Guideline to Files

`main.py`:
Launches the overall Federated Learning program

`configurations.py`:
Initialises background classes and settings

`flgnn_dataset.py`:
Includes all gnn data loading and splitting functions

`graph_partition.py`:
Implements our graph partitioning algorithm

`fl_clients.py`:
Creates a basic gnn client class, and list out all client-related functions

`fl_strategy.py`:
Runs the Federated Learning process, called by `main.py`

---

# Downloading Dataset

Due to the large sizes of the datasets, we include the link to download each dataset in a [Google Drive](https://drive.google.com/drive/folders/19BWid2En9IWdzbPeZ3Tj29c4iDdXhtRV?usp=drive_link). Simply download the fils and copy them into the directory `/data/`

---

# Building Our Graph Partitioning Algorithm

Our Graph Partitioning Algorithm `CoLearnPartition` is written in C++ for efficiency. To run this, you need to build the C++ file using:

```
source build_gpa.sh
```

---

# Running Code

We can run the program using:
`python3 main.py $dataset$`

where
`dataset`:  datasets, options are {bitcoinOTC, UCI, DBLP3, DBLP5, Reddit, SBM}

Example Experiments

`python3 main.py bitcoinOTC`
`python3 main.py DBLP3`
`python3 main.py Reddit`  
`python3 main.py SBM`

---

# Plotting and Analysis

You can visualise the result using `analysis_gpa.ipynb` for comparing graph partitioning algorithms; and `analysis_ne.ipynb` for comparing node embedding exchange schemes.

Simply change the path of the log result recorded using our logging system and rename it to your desire.

---
