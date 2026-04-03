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

# Step 1: Downloading Dataset

Due to the large sizes of the node classification datasets, we include the link to download each dataset in a [Google Drive](https://drive.google.com/drive/folders/19BWid2En9IWdzbPeZ3Tj29c4iDdXhtRV?usp=drive_link). Simply download the files and copy them into the `data/` directory

---

# Step 2: Constructing Environment

## Recommended: Conda environment (works without sudo)

Some packages in `requirements.txt` are pinned to older versions (for example `pandas==1.3.5`) and may fail with newer default Python installations. We recommend using a dedicated Conda environment:

```
conda create -y -n dgcolearn311 python=3.11
conda activate dgcolearn311

conda install -y numpy=1.25.2 scipy=1.9.3 scikit-learn=1.5.0
python -m pip install --upgrade "pip<25" "setuptools<81" "wheel<0.45" "Cython<3"
python -m pip install --no-build-isolation pandas==1.3.5
python -m pip install --no-build-isolation -r requirements.txt
```

If `conda` is unavailable on your machine, install [Miniforge](https://github.com/conda-forge/miniforge) in your home directory, then run the same commands.

## Alternative: Python venv

If you already have a compatible `python3.11` available:

```
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

# Step 3: Building Our Graph Partitioning Algorithm

Our Graph Partitioning Algorithm `CoLearnPartition` is written in C++ for efficiency. To run this, you need to build the C++ file using:

```
source build_gpa.sh
```

---

# Step 4: Running Code

We can run the program using:

```
python3 main.py $dataset$
```

where
`dataset`:  datasets, options are {bitcoinOTC, UCI, DBLP3, DBLP5, Reddit, as733, tgbl-comment}

Example Experiments

`python3 main.py bitcoinOTC`
`python3 main.py DBLP3`
`python3 main.py Reddit`
`python3 main.py tgbl-comment`

The configuration of AS-733 is different, we have provided a configuration file `as733.yaml` as another argument.

`python3 main.py as733 as733.yaml`

##### Running Repeated Experiments

With the following command, we can run multiple experiments.

```
bash ./run_repeated_job.sh
```

##### Ablation Study

You can also run the traditional full-graph training (instead of incremental learning) by setting the argument `--incremental_learning` to False:

```
python3 main.py $dataset$ --incremental_learning False
```

(The default value of this argument is `True`)

---

# Plotting and Analysis

You can visualise the result using `analysis_gpa.ipynb` for comparing graph partitioning algorithms; and `analysis_ne.ipynb` for comparing node embedding exchange schemes.

Simply change the path of the log result recorded using our logging system and rename it to your desire.

---
